import logging
import multiprocessing
import os
import tempfile
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from shutil import rmtree

from . import pickling
from .multiprocessing_logging_handler import get_multiprocessing_logging_setup_fn
from .schedulers.kube import KubernetesExecutor
from .schedulers.pbs import PBSExecutor
from .schedulers.slurm import SlurmExecutor
from .util import enrich_future_with_uncaught_warning


def get_existent_kwargs_subset(whitelist, kwargs):
    new_kwargs = {}
    for arg_name in whitelist:
        if arg_name in kwargs:
            new_kwargs[arg_name] = kwargs[arg_name]

    return new_kwargs


PROCESS_POOL_KWARGS_WHITELIST = ["max_workers", "initializer", "initargs"]


class WrappedProcessPoolExecutor(ProcessPoolExecutor):
    """
    Wraps the ProcessPoolExecutor to add various features:
    - map_to_futures and map_unordered method
    - pickling of job's output (see output_pickle_path_getter and output_pickle_path)
    - job submission via pickling to circumvent bug in python < 3.8 (see MULTIPROCESSING_VIA_IO_TMP_DIR)
    """

    def __init__(self, **kwargs):
        assert (not "start_method" in kwargs or kwargs["start_method"] is None) or (
            not "mp_context" in kwargs
        ), "Cannot use both `start_method` and `mp_context` kwargs."

        new_kwargs = get_existent_kwargs_subset(PROCESS_POOL_KWARGS_WHITELIST, kwargs)

        mp_context = None

        if "mp_context" in kwargs:
            mp_context = kwargs["mp_context"]
        elif "start_method" in kwargs and kwargs["start_method"] is not None:
            mp_context = multiprocessing.get_context(kwargs["start_method"])
        elif "MULTIPROCESSING_DEFAULT_START_METHOD" in os.environ:
            mp_context = multiprocessing.get_context(
                os.environ["MULTIPROCESSING_DEFAULT_START_METHOD"]
            )
        else:
            mp_context = multiprocessing.get_context("spawn")

        new_kwargs["mp_context"] = mp_context

        ProcessPoolExecutor.__init__(self, **new_kwargs)

    def submit(self, *args, **kwargs):

        output_pickle_path = None
        if "__cfut_options" in kwargs:
            output_pickle_path = kwargs["__cfut_options"]["output_pickle_path"]
            del kwargs["__cfut_options"]

        if os.environ.get("MULTIPROCESSING_VIA_IO"):
            # If MULTIPROCESSING_VIA_IO is set, _submit_via_io is used to
            # workaround size constraints in pythons multiprocessing
            # implementation. Also see https://github.com/python/cpython/pull/10305/files
            # This should be fixed in python 3.8
            submit_fn = self._submit_via_io
        else:
            submit_fn = super().submit

        # Depending on the start_method and output_pickle_path, wrapper functions may need to be
        # executed in the new process context, before the actual code is ran.
        # These wrapper functions consume their arguments from *args, **kwargs and assume
        # that the next argument will be another function that is then called.
        # The call_stack holds all of these wrapper functions and their arguments in the correct order.
        # For example, call_stack = [wrapper_fn_1, wrapper_fn_1_arg_1, wrapper_fn_2, actual_fn, actual_fn_arg_1]
        # where wrapper_fn_1 is called, which eventually calls wrapper_fn_2, which eventually calls actual_fn.
        call_stack = []

        if self._mp_context.get_start_method() != "fork":
            # If a start_method other than the default "fork" is used, logging needs to be re-setup,
            # because the programming context is not inherited in those cases.
            multiprocessing_logging_setup_fn = get_multiprocessing_logging_setup_fn()
            call_stack.extend(
                [
                    WrappedProcessPoolExecutor._setup_logging_and_execute,
                    multiprocessing_logging_setup_fn,
                ]
            )

        if output_pickle_path is not None:
            call_stack.extend(
                [
                    WrappedProcessPoolExecutor._execute_and_persist_function,
                    output_pickle_path,
                ]
            )

        fut = submit_fn(*call_stack, *args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def _submit_via_io(self, *args, **kwargs):

        func = args[0]
        args = args[1:]

        opt_tmp_dir = os.environ.get("MULTIPROCESSING_VIA_IO_TMP_DIR")
        if opt_tmp_dir is not None:
            dirpath = tempfile.mkdtemp(dir=opt_tmp_dir)
        else:
            dirpath = tempfile.mkdtemp()

        output_pickle_path = Path(dirpath) / "jobdescription.pickle"

        with open(output_pickle_path, "wb") as file:
            pickling.dump((func, args, kwargs), file)

        future = super().submit(
            WrappedProcessPoolExecutor._execute_via_io, output_pickle_path
        )

        future.add_done_callback(
            partial(WrappedProcessPoolExecutor._remove_tmp_file, dirpath)
        )

        return future

    @staticmethod
    def _remove_tmp_file(path, _future):
        rmtree(path)

    @staticmethod
    def _setup_logging_and_execute(multiprocessing_logging_setup_fn, *args, **kwargs):

        func = args[0]
        args = args[1:]

        multiprocessing_logging_setup_fn()

        return func(*args, **kwargs)

    @staticmethod
    def _execute_via_io(serialized_function_info_path):

        with open(serialized_function_info_path, "rb") as file:
            (func, args, kwargs) = pickling.load(file)
        return func(*args, **kwargs)

    @staticmethod
    def _execute_and_persist_function(output_pickle_path, *args, **kwargs):

        func = args[0]
        args = args[1:]

        try:
            result = True, func(*args, **kwargs)
        except Exception as exc:
            result = False, exc
            logging.warning(f"Job computation failed with:\n{exc.__repr__()}")

        with open(output_pickle_path, "wb") as file:
            pickling.dump(result, file)

        if result[0]:
            return result[1]
        else:
            raise result[1]

    def map_unordered(self, func, args):

        futs = self.map_to_futures(func, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator():
            for fut in futures.as_completed(futs):
                yield fut.result()

        return result_generator()

    def map_to_futures(self, func, args, output_pickle_path_getter=None):

        if output_pickle_path_getter is not None:
            futs = [
                self.submit(
                    func,
                    arg,
                    __cfut_options={
                        "output_pickle_path": output_pickle_path_getter(arg)
                    },
                )
                for arg in args
            ]
        else:
            futs = [self.submit(func, arg) for arg in args]

        return futs

    def forward_log(self, fut):
        """
        Similar to the cluster executor, this method Takes a future from which the log file is forwarded to the active
        process. This method blocks as long as the future is not done.
        """

        # Since the default behavior of process pool executors is to show the log in the main process
        # we don't need to do anything except for blocking until the future is done.
        return fut.result()


class SequentialExecutor(WrappedProcessPoolExecutor):
    """
    The same as WrappedProcessPoolExecutor, but always uses only one core. In essence,
    this is a sequential executor approach, but it still makes use of the standard pool approach.
    That way, switching between different executors should always work without any problems.
    """

    def __init__(self, **kwargs):
        kwargs["max_workers"] = 1
        WrappedProcessPoolExecutor.__init__(self, **kwargs)


class DebugSequentialExecutor(SequentialExecutor):
    """
    Only use for debugging purposes. This executor does not spawn new processes for its jobs. Therefore,
    setting breakpoint()'s should be possible without context-related problems.
    """

    def submit(self, *args, **kwargs):

        output_pickle_path = None
        if "__cfut_options" in kwargs:
            output_pickle_path = kwargs["__cfut_options"]["output_pickle_path"]
            del kwargs["__cfut_options"]

        if output_pickle_path is not None:
            fut = self._blocking_submit(
                WrappedProcessPoolExecutor._execute_and_persist_function,
                output_pickle_path,
                *args,
                **kwargs,
            )
        else:
            fut = self._blocking_submit(*args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def _blocking_submit(self, *args, **kwargs):

        func = args[0]
        args = args[1:]

        fut = futures.Future()
        result = func(*args, **kwargs)
        fut.set_result(result)

        return fut


def pickle_identity(obj):
    return pickling.loads(pickling.dumps(obj))


def pickle_identity_executor(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return pickle_identity(result)


class PickleExecutor(WrappedProcessPoolExecutor):
    """
    The same as WrappedProcessPoolExecutor, but always pickles input and output of the jobs.
    When using this executor for automated tests, it is ensured that using cluster executors in production
    won't provoke pickling-related problems.
    """

    def submit(self, _func, *_args, **_kwargs):

        (func, args, kwargs) = pickle_identity((_func, _args, _kwargs))
        return super().submit(pickle_identity_executor, func, *args, **kwargs)


def noop():
    return True


did_start_test_multiprocessing = False


def test_valid_multiprocessing():

    msg = """
    ###############################################################
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            main()
            ...
    ###############################################################
    """

    with get_executor("multiprocessing") as executor:
        try:
            res_fut = executor.submit(noop)
            assert res_fut.result() == True, msg
        except RuntimeError as exc:
            raise Exception(msg) from exc
        except EOFError as exc:
            raise Exception(msg) from exc


def get_executor(environment, **kwargs):

    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "pbs":
        return PBSExecutor(**kwargs)
    elif environment == "kubernetes":
        return KubernetesExecutor(**kwargs)
    elif environment == "multiprocessing":
        global did_start_test_multiprocessing
        if not did_start_test_multiprocessing:
            did_start_test_multiprocessing = True
            test_valid_multiprocessing()

        return WrappedProcessPoolExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
    elif environment == "debug_sequential":
        return DebugSequentialExecutor(**kwargs)
    elif environment == "test_pickling":
        return PickleExecutor(**kwargs)
    raise Exception("Unknown executor: {}".format(environment))

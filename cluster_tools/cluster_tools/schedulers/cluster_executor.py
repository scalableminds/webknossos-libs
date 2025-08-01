import logging
import os
import signal
import sys
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from concurrent import futures
from concurrent.futures import Future
from functools import partial
from types import FrameType, TracebackType
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from typing_extensions import ParamSpec

from cluster_tools._utils import pickling
from cluster_tools._utils.file_wait_thread import FileWaitThread
from cluster_tools._utils.reflection import (
    file_path_to_absolute_module,
    get_function_name,
)
from cluster_tools._utils.string_ import random_string, with_preliminary_postfix
from cluster_tools._utils.tailf import Tail
from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict

NOT_YET_SUBMITTED_STATE_TYPE = Literal["NOT_YET_SUBMITTED"]
NOT_YET_SUBMITTED_STATE: NOT_YET_SUBMITTED_STATE_TYPE = "NOT_YET_SUBMITTED"

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


def join_messages(strings: list[str]) -> str:
    return " ".join(x.strip() for x in strings if x.strip())


class RemoteException(Exception):
    def __init__(self, error: str, job_id: str):
        self.error = error
        self.job_id = job_id

    def __str__(self) -> str:
        return self.error.strip() + f" (job_id={self.job_id})"


class RemoteResourceLimitException(RemoteException):
    def __str__(self) -> str:
        return str(self.job_id) + "\n" + self.error.strip()


class RemoteOutOfMemoryException(RemoteResourceLimitException):
    pass


class RemoteTimeLimitException(RemoteResourceLimitException):
    pass


class ClusterExecutor(futures.Executor):
    """Futures executor for executing jobs on a cluster."""

    _shutdown_hooks: list[Callable[[], None]] = []
    _installed_signal_handler: bool = False

    def __init__(
        self,
        debug: bool = False,
        keep_logs: bool = True,
        cfut_dir: str | None = None,
        job_resources: dict[str, Any] | None = None,
        job_name: str | None = None,
        additional_setup_lines: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        `kwargs` can be the following optional parameters:
            `logging_config`: An object containing a `level` key specifying the desired log level and/or a
                `format` key specifying the desired log format string. Cannot be specified together
                with `logging_setup_fn`.
            `logging_setup_fn`: A function setting up custom logging. The function will be called for
                remotely executed code (slurm, pbs) to re-setup logging. The function will be called with the
                default log file name. If the caller sets up file logging, this log file name should be adapted,
                for example, by adding a .mylog suffix. Cannot be specified together with `logging_config`.
        """
        self.debug = debug
        self.job_resources = job_resources
        self.additional_setup_lines = additional_setup_lines or []
        self.job_name = job_name
        self.cfut_dir = (
            cfut_dir if cfut_dir is not None else os.getenv("CFUT_DIR", ".cfut")
        )
        self.files_to_clean_up: list[str] = []

        logging.info(
            f"Instantiating ClusterExecutor. Log files are stored in {self.cfut_dir}"
        )

        # `jobs` maps from job id to (future, workerid, outfile_name, should_keep_output)
        # In case, job arrays are used: job id and workerid are in the format of
        # `job_id-job_index` and `workerid-job_index`.
        self.jobs: dict[
            str,
            NOT_YET_SUBMITTED_STATE_TYPE | tuple[Future, str, str, bool],
        ] = {}
        self.jobs_lock = threading.Lock()
        self.jobs_empty_cond = threading.Condition(self.jobs_lock)
        self.keep_logs = keep_logs
        self.is_shutting_down = False

        self.wait_thread = FileWaitThread(self._completion, self)
        self.wait_thread.start()

        os.makedirs(self.cfut_dir, exist_ok=True)

        self._register_shutdown_hook(self.handle_kill)

        self.metadata = {}
        assert not ("logging_config" in kwargs and "logging_setup_fn" in kwargs), (
            "Specify either logging_config OR logging_setup_fn but not both at once"
        )
        if "logging_config" in kwargs:
            self.metadata["logging_config"] = kwargs["logging_config"]
        if "logging_setup_fn" in kwargs:
            self.metadata["logging_setup_fn"] = kwargs["logging_setup_fn"]

    @classmethod
    def as_completed(cls, futs: list[Future[_T]]) -> Iterator[Future[_T]]:
        return futures.as_completed(futs)

    @classmethod
    @abstractmethod
    def executor_key(cls) -> str:
        pass

    @classmethod
    def _ensure_signal_handlers_are_installed(cls) -> None:
        # Only overwrite the signal handler once
        if cls._installed_signal_handler:
            return

        # Clean up if a SIGINT or SIGTERM signal is received. However, do not
        # interfere with the existing signal handler of the process and execute
        # it afterwards.
        existing_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(
            signal.SIGINT,
            partial(cls._handle_shutdown, existing_sigint_handler),
        )
        existing_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(
            signal.SIGTERM,
            partial(cls._handle_shutdown, existing_sigterm_handler),
        )

        cls._installed_signal_handler = True

    @classmethod
    def _register_shutdown_hook(cls, hook: Callable[[], None]) -> None:
        cls._shutdown_hooks.append(hook)
        cls._ensure_signal_handlers_are_installed()

    @classmethod
    def _deregister_shutdown_hook(cls, hook: Callable[[], None]) -> None:
        if hook in cls._shutdown_hooks:
            cls._shutdown_hooks.remove(hook)
        else:
            logging.warning(
                "Cannot deregister executors shutdown hook since it's not registered."
            )

    @classmethod
    def _handle_shutdown(
        cls,
        existing_signal_handler: Callable[[int, FrameType | None], None] | int | None,
        signum: int,
        frame: Any,
    ) -> None:
        logging.critical(
            f"[{cls.__name__}] Caught signal {signal.Signals(signum).name}, running shutdown hooks"
        )
        try:
            for hook in cls._shutdown_hooks:
                hook()
        except Exception as e:
            print(f"Error during shutdown: {e}")

        if (
            callable(existing_signal_handler)
            and existing_signal_handler
            not in (
                signal.SIG_DFL,  # For completeness sake (since it's not callable anyways). The system's default signal handler
                signal.SIG_IGN,  # For completeness sake (since it's not callable anyways). The instruction to ignore a signal
                signal.default_int_handler,  # Python's default SIGINT handler
            )
        ):
            existing_signal_handler(signum, frame)

    def handle_kill(self) -> None:
        if self.is_shutting_down:
            return

        self.is_shutting_down = True

        self.wait_thread.stop()
        self.inner_handle_kill()
        self.clean_up()

    @abstractmethod
    def inner_handle_kill(self) -> None:
        pass

    @abstractmethod
    def check_job_state(
        self, job_id_with_index: str
    ) -> Literal["failed", "ignore", "completed"]:
        pass

    @staticmethod
    @abstractmethod
    def get_current_job_id() -> str:
        pass

    def investigate_failed_job(
        self,
        job_id_with_index: str,  # noqa: ARG002 Unused method argument: `job_id_with_index`
    ) -> tuple[str, type[RemoteException]] | None:
        """
        When a job fails, this method is called to investigate why. If a tuple is returned,
        the containing message (1st element) will be attached to the thrown exception (which will use
        the returned exception type from the 2nd element).
        For example, this method could be used to check for common problems, such as violated
        RAM constraints.
        """
        return None

    def _start(
        self,
        workerid: str,
        job_count: int | None = None,
        job_name: str | None = None,
    ) -> tuple[list[Future[str]], list[tuple[int, int]]]:
        """Start job(s) with the given worker ID and return IDs
        identifying the new job(s). The job should run ``python -m
        cfut.remote <executorkey> <workerid>.
        """

        jobids_futures, job_index_ranges = self.inner_submit(
            f"{self.get_python_executable()} -m cluster_tools.remote {self.executor_key()} {workerid} {self.cfut_dir}",
            job_name=self.job_name if self.job_name is not None else job_name,
            additional_setup_lines=self.additional_setup_lines,
            job_count=job_count,
        )

        # Since not all jobs may be submitted immediately, cluster executors return
        # jobid futures in the inner_submit function. Also, since it may not be allowed
        # to submit all jobs at once, the jobs may be submitted in batches with each batch
        # containing a subset of all jobs identified by a separate jobid. The job_index_ranges
        # array of (start_index, end_index) tuples, indicates which of the job_count
        # jobs were submitted in each batch. start_index is inclusive whereas end_index
        # is not.
        return jobids_futures, job_index_ranges

    @abstractmethod
    def inner_submit(
        self,
        cmdline: str,
        job_name: str | None = None,
        additional_setup_lines: list[str] | None = None,
        job_count: int | None = None,
    ) -> tuple[list[Future[str]], list[tuple[int, int]]]:
        pass

    def _maybe_mark_logs_for_cleanup(self, jobid: str) -> None:
        """Given a job ID as returned by _start, perform any necessary
        cleanup after the job has finished.
        """
        if self.keep_logs:
            return

        log_path = self.format_log_file_path(self.cfut_dir, jobid)
        self.files_to_clean_up.append(log_path)

    @staticmethod
    @abstractmethod
    def format_log_file_name(job_id_with_index: str, suffix: str = ".stdout") -> str:
        pass

    @classmethod
    def format_log_file_path(
        cls, cfut_dir: str, job_id_with_index: str, suffix: str = ".stdout"
    ) -> str:
        return os.path.join(
            cfut_dir, cls.format_log_file_name(job_id_with_index, suffix)
        )

    @classmethod
    @abstractmethod
    def get_job_id_string(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_job_array_index() -> int | None:
        pass

    @staticmethod
    def get_temp_file_path(cfut_dir: str, file_name: str) -> str:
        return os.path.join(cfut_dir, file_name)

    @staticmethod
    def format_infile_name(cfut_dir: str, job_id: str) -> str:
        return os.path.join(cfut_dir, f"cfut.in.{job_id}.pickle")

    @staticmethod
    def format_outfile_name(cfut_dir: str, job_id: str) -> str:
        return os.path.join(cfut_dir, f"cfut.out.{job_id}.pickle")

    def get_python_executable(self) -> str:
        return sys.executable

    def _completion(self, jobid: str, failed_early: bool) -> None:
        """Called whenever a job finishes."""
        with self.jobs_lock:
            job_info = self.jobs.pop(jobid)
            assert job_info != NOT_YET_SUBMITTED_STATE

            if len(job_info) == 4:
                fut, workerid, outfile_name, should_keep_output = job_info
            else:
                # Backwards compatibility
                fut, workerid = job_info  # type: ignore[misc]
                should_keep_output = False
                outfile_name = self.format_outfile_name(self.cfut_dir, workerid)

            if not self.jobs:
                self.jobs_empty_cond.notify_all()
        if self.debug:
            logging.debug(f"Job completed: {jobid}")

        preliminary_outfile_name = with_preliminary_postfix(outfile_name)

        # By default, exceptions are wrapped by the RemoteException class.
        # However, this can be customized by investigating the actual error
        # using `investigate_failed_job`.
        wrapping_exception_cls = RemoteException
        if failed_early:
            # If the job failed, but didn't write the error to an output file,
            # we handle this case separately.
            # Typical reasons could be for this:
            # - because python isn't installed or the cluster_tools couldn't be found
            # - because the job was killed (e.g., by slurm due to RAM limit violations)
            # We don't try to deserialize pickling output, because it won't exist.
            success = False

            opt_reason_and_exception_cls = self.investigate_failed_job(jobid)
            reason = None
            if opt_reason_and_exception_cls is not None:
                reason, wrapping_exception_cls = opt_reason_and_exception_cls

            result = join_messages(
                [
                    "Job submission/execution failed.",
                    reason or "",
                    f"Please look into the log file at {self.format_log_file_path(self.cfut_dir, jobid)}.",
                ]
            )
        else:
            with open(preliminary_outfile_name, "rb") as f:
                outdata = f.read()
            success, result = pickling.loads(outdata)

        if success:
            # Remove the .preliminary postfix since the job was finished
            # successfully. Therefore, the result can be used as a checkpoint
            # by users of the clustertools.
            os.rename(preliminary_outfile_name, outfile_name)
            logging.debug(f"Pickle file renamed to {outfile_name}.")

            fut.set_result(result)
        else:
            # Don't remove the .preliminary postfix since the job failed.
            remote_exc = wrapping_exception_cls(result, jobid)
            fut.set_exception(remote_exc)

        # Clean up communication files.

        infile_name = self.format_infile_name(self.cfut_dir, workerid)
        self.files_to_clean_up.append(infile_name)
        if not should_keep_output:
            self.files_to_clean_up.append(outfile_name)

        self._maybe_mark_logs_for_cleanup(jobid)

    def ensure_not_shutdown(self) -> None:
        if self.is_shutting_down:
            raise RuntimeError(
                "submit() was invoked on a ClusterExecutor instance even though shutdown() or handle_kill() was executed for that instance."
            )

    def create_enriched_future(self) -> Future:
        fut: Future = Future()
        enrich_future_with_uncaught_warning(fut)
        return fut

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        """
        Submit a job to the pool.
        kwargs may contain __cfut_options which currently should look like:
        {
            "output_pickle_path": str
        }
        output_pickle_path defines where the pickled result should be stored.
        That file will not be removed after the job has finished.
        """
        fut = self.create_enriched_future()
        workerid = random_string()

        if "__cfut_options" in kwargs:
            should_keep_output = True
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]
        else:
            should_keep_output = False
            output_pickle_path = self.format_outfile_name(self.cfut_dir, workerid)

        self.ensure_not_shutdown()

        # Start the job.
        serialized_function_info = pickling.dumps(
            ((__fn, self.metadata), args, kwargs, output_pickle_path)
        )
        with open(self.format_infile_name(self.cfut_dir, workerid), "wb") as f:
            f.write(serialized_function_info)

        self.store_main_path_to_meta_file(workerid)

        preliminary_output_pickle_path = with_preliminary_postfix(output_pickle_path)
        if os.path.exists(preliminary_output_pickle_path):
            logging.warning(
                f"Deleting stale output file at {preliminary_output_pickle_path}..."
            )
            os.unlink(preliminary_output_pickle_path)

        job_name = get_function_name(__fn)
        jobids_futures, _ = self._start(workerid, job_name=job_name)
        # Only a single job was submitted
        jobid = jobids_futures[0].result()

        if self.debug:
            logging.debug(f"Job submitted: {jobid}")

        # Thread will wait for it to finish.
        self.wait_thread.waitFor(preliminary_output_pickle_path, jobid)

        with self.jobs_lock:
            self.jobs[jobid] = (fut, workerid, output_pickle_path, should_keep_output)

        fut.cluster_jobid = jobid  # type: ignore[attr-defined]
        return fut

    @classmethod
    def get_workerid_with_index(cls, workerid: str, index: int | str) -> str:
        return f"{workerid}_{index}"

    @classmethod
    def get_jobid_with_index(cls, jobid: str | int, index: int) -> str:
        return f"{jobid}_{index}"

    def get_function_and_metadata_pickle_path(self, workerid: str) -> str:
        return self.format_infile_name(
            self.cfut_dir,
            self.get_workerid_with_index(workerid, "function-and-metadata"),
        )

    @staticmethod
    def get_main_meta_path(cfut_dir: str, workerid: str) -> str:
        return os.path.join(cfut_dir, f"cfut.main_path.{workerid}.txt")

    def store_main_path_to_meta_file(self, workerid: str) -> None:
        main_meta_path = self.get_main_meta_path(self.cfut_dir, workerid)
        with open(main_meta_path, "w", encoding="utf-8") as file:
            file.write(file_path_to_absolute_module(sys.argv[0]))
        self.files_to_clean_up.append(main_meta_path)

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[
            _S
        ],  # TODO change: allow more than one arg per call # noqa FIX002 Line contains TODO
        output_pickle_path_getter: Callable[[_S], os.PathLike] | None = None,
    ) -> list[Future[_T]]:
        self.ensure_not_shutdown()
        args = list(args)
        if len(args) == 0:
            return []

        should_keep_output = output_pickle_path_getter is not None

        futs_with_output_paths = []
        workerid = random_string()

        pickled_function_and_metadata_path = self.get_function_and_metadata_pickle_path(
            workerid
        )
        self.files_to_clean_up.append(pickled_function_and_metadata_path)
        with open(pickled_function_and_metadata_path, "wb") as file:
            pickling.dump((fn, self.metadata), file)
        self.store_main_path_to_meta_file(workerid)

        for index, arg in enumerate(args):
            fut = self.create_enriched_future()
            workerid_with_index = self.get_workerid_with_index(workerid, index)

            if output_pickle_path_getter is None:
                output_pickle_path = self.format_outfile_name(
                    self.cfut_dir, workerid_with_index
                )
            else:
                output_pickle_path = str(output_pickle_path_getter(arg))

            preliminary_output_pickle_path = with_preliminary_postfix(
                output_pickle_path
            )
            if os.path.exists(preliminary_output_pickle_path):
                logging.warning(
                    f"Deleting stale output file at {preliminary_output_pickle_path}..."
                )
                os.unlink(preliminary_output_pickle_path)

            serialized_function_info = pickling.dumps(
                (
                    pickled_function_and_metadata_path,
                    [arg],
                    {},
                    output_pickle_path,
                )
            )
            infile_name = self.format_infile_name(self.cfut_dir, workerid_with_index)

            with open(infile_name, "wb") as f:
                f.write(serialized_function_info)

            futs_with_output_paths.append((fut, output_pickle_path))

        with self.jobs_lock:
            # Use a separate loop to avoid having to acquire the jobs_lock many times
            # or for the full duration of the above loop
            for index in range(len(futs_with_output_paths)):
                workerid_with_index = self.get_workerid_with_index(workerid, index)
                # Register the job in the jobs array, although the jobid is not known yet.
                # Otherwise it might happen that self.jobs becomes empty, but some of the jobs were
                # not even submitted yet.
                self.jobs[workerid_with_index] = NOT_YET_SUBMITTED_STATE

        job_count = len(args)
        job_name = get_function_name(fn)
        jobids_futures, job_index_ranges = self._start(workerid, job_count, job_name)

        number_of_batches = len(jobids_futures)
        for batch_index, (jobid_future, (job_index_start, job_index_end)) in enumerate(
            zip(jobids_futures, job_index_ranges)
        ):
            jobid_future.add_done_callback(
                partial(
                    self.register_jobs,
                    futs_with_output_paths[job_index_start:job_index_end],
                    workerid,
                    should_keep_output,
                    job_index_start,
                    f"{batch_index + 1}/{number_of_batches}",
                )
            )

        return [fut for (fut, _) in futs_with_output_paths]

    def register_jobs(
        self,
        futs_with_output_paths: list[tuple[Future, str]],
        workerid: str,
        should_keep_output: bool,
        job_index_offset: int,
        batch_description: str,
        jobid_future: Future[str],
    ) -> None:
        jobid = jobid_future.result()
        if self.debug:
            logging.debug(
                f"Submitted array job {batch_description} with JobId {jobid} and {len(futs_with_output_paths)} subjobs.",
            )

        for array_index, (fut, output_path) in enumerate(futs_with_output_paths):
            jobid_with_index = self.get_jobid_with_index(jobid, array_index)

            # Thread will wait for it to finish.
            self.wait_thread.waitFor(
                with_preliminary_postfix(output_path), jobid_with_index
            )

            fut.cluster_jobid = jobid  # type: ignore[attr-defined]
            # fut.cluster_jobindex is only used for debugging:
            fut.cluster_jobindex = array_index  # type: ignore[attr-defined]

            job_index = job_index_offset + array_index
            workerid_with_index = self.get_workerid_with_index(workerid, job_index)
            # Remove the pending jobs entry and add the correct one
            with self.jobs_lock:
                del self.jobs[workerid_with_index]
                self.jobs[jobid_with_index] = (
                    fut,
                    workerid_with_index,
                    output_path,
                    should_keep_output,
                )

    # Overwrite the context manager __exit as it doesn't forward the information whether an exception was thrown or not otherwise
    # which may lead to a deadlock if an exception is thrown within a cluster executor with statement, because self.jobs_empty_cond.wait()
    # never succeeds.
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> Literal[False]:
        # Don't wait if an exception was thrown
        self.shutdown(wait=exc_type is None)
        return False

    def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
        """Close the pool."""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        if not cancel_futures:
            logging.warning(
                "The provided cancel_futures argument is ignored by ClusterExecutor."
            )
        if wait:
            with self.jobs_lock:
                if self.jobs and self.wait_thread.is_alive():
                    self.jobs_empty_cond.wait()
        else:
            self.inner_handle_kill()

        self.wait_thread.stop()
        self.wait_thread.join()

        self.clean_up()

    def clean_up(self) -> None:
        for file_to_clean_up in self.files_to_clean_up:
            try:
                os.unlink(file_to_clean_up)
            except OSError as exc:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
                logging.warning(
                    f"Could not delete file during clean up. Path: {file_to_clean_up} Exception: {exc}. Continuing..."
                )
        self.files_to_clean_up = []
        self._deregister_shutdown_hook(self.handle_kill)

    def map(  # type: ignore[override]
        self,
        fn: Callable[[_S], _T],
        args: Iterable[Any],
        timeout: float | None = None,
        chunksize: int | None = None,
    ) -> Iterator[_T]:
        if chunksize is not None:
            logging.warning(
                "The provided chunksize argument is ignored by ClusterExecutor."
            )

        start_time = time.time()

        futs = self.map_to_futures(fn, args)

        # Return a separate generator as an iterator to avoid that the
        # map() method itself becomes a generator (due to the usage of yield).
        # If map() was a generator, the submit() calls would be invoked
        # lazily which can lead to a shutdown of the executor before
        # the submit calls are performed.
        def result_generator() -> Iterator[_T]:
            for fut in futs:
                passed_time = time.time() - start_time
                remaining_timeout = None if timeout is None else timeout - passed_time
                yield fut.result(remaining_timeout)

        return result_generator()

    def forward_log(self, fut: Future[_T]) -> _T:
        """
        Takes a future from which the log file is forwarded to the active
        process. This method blocks as long as the future is not done.
        """

        log_path = self.format_log_file_path(self.cfut_dir, fut.cluster_jobid)  # type: ignore[attr-defined]

        # Don't use a logger instance here, since the child process
        # probably already used a logger.
        def log_callback(s: str) -> None:
            sys.stdout.write(f"(jid={fut.cluster_jobid}) {s}")  # type: ignore[attr-defined]

        tailer = Tail(log_path, log_callback)
        fut.add_done_callback(lambda _: tailer.cancel())

        # Poll until the log file exists
        while not (os.path.exists(log_path) or tailer.is_cancelled):
            time.sleep(2)

        # Log the output of the log file until future is resolved
        # by the done_callback we attached earlier.
        tailer.follow(2)
        return fut.result()

    @abstractmethod
    def get_pending_tasks(self) -> Iterable[str]:
        pass

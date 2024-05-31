import logging
import os
import re
import signal
from concurrent import futures
from concurrent.futures import Future
from functools import partial
from multiprocessing import Queue, get_context
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    cast,
)
from weakref import ReferenceType, ref

from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict, MultiprocessingExecutor

if TYPE_CHECKING:
    from distributed import Client

logger = logging.getLogger()


_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


def _run_in_nanny(
    queue: Queue, __fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> None:
    try:
        __env = cast(Dict[str, str], kwargs.pop("__env"))
        for key, value in __env.items():
            os.environ[key] = value

        if "PWD" in os.environ:
            os.chdir(os.environ["PWD"])
        ret = __fn(*args, **kwargs)
        queue.put({"value": ret})
    except Exception as exc:
        queue.put({"exception": exc})


def _run_with_nanny(
    __fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    mp_context = get_context("spawn")
    q = mp_context.Queue()
    p = mp_context.Process(target=_run_in_nanny, args=(q, __fn) + args, kwargs=kwargs)
    p.start()
    p.join()
    ret = q.get(timeout=0.1)
    if "exception" in ret:
        raise ret["exception"]
    else:
        return ret["value"]


def _parse_mem(size: str) -> int:
    units = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    m = re.match(r"^([\d\.]+)\s*([kmgtKMGT]{0,1})$", str(size).strip())
    assert m is not None, f"Could not parse {size}"
    number, unit = float(m.group(1)), m.group(2).upper()
    assert unit in units
    return int(number * units[unit])


def _handle_kill_through_weakref(
    executor_ref: "ReferenceType[DaskExecutor]",
    existing_sigint_handler: Any,
    signum: Optional[int],
    frame: Any,
) -> None:
    executor = executor_ref()
    if executor is None:
        return
    executor.handle_kill(existing_sigint_handler, signum, frame)


class DaskExecutor(futures.Executor):
    """
    The `DaskExecutor` allows to run workloads on a dask cluster.

    The executor can be constructed with an existing dask `Client` or
    from a declarative configuration. The address of the dask scheduler
    can be part of the configuration or supplied as environment variable
    `DASK_ADDRESS`.

    There is support for resource-based scheduling. As default, `mem` and
    `cpus-per-task` are supported. To make use of them, the dask workers
    should be started with:
    `python -m dask worker --no-nanny --nthreads 6 tcp://... --resources "mem=1073741824 cpus=8"`
    """

    client: "Client"
    pending_futures: Set[Future]
    job_resources: Optional[Dict[str, Any]]
    is_shutting_down = False

    def __init__(
        self, client: "Client", job_resources: Optional[Dict[str, Any]] = None
    ) -> None:
        try:
            import distributed  # noqa: F401 unused import
        except ModuleNotFoundError:
            logger.error(
                'The distributed Python package for Dask is not installed. cluster_tools does not install this dependency be default. Run `pip install cluster_tools[dask]` or `poetry install --extras "dask"` to install Dask support.'
            )
            exit()

        self.client = client
        self.pending_futures = set()
        self.job_resources = job_resources

        if self.job_resources is not None:
            # `mem` needs to be a number for dask, so we need to parse it
            if "mem" in self.job_resources:
                self.job_resources["mem"] = _parse_mem(self.job_resources["mem"])
            if "cpus-per-task" in self.job_resources:
                self.job_resources["cpus"] = int(
                    self.job_resources.pop("cpus-per-task")
                )

        # Clean up if a SIGINT signal is received. However, do not interfere with the
        # existing signal handler of the process or the
        # shutdown of the main process which sends SIGTERM signals to terminate all
        # child processes.
        existing_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(
            signal.SIGINT,
            partial(_handle_kill_through_weakref, ref(self), existing_sigint_handler),
        )

    @classmethod
    def from_config(
        cls,
        job_resources: Dict[str, str],
        **_kwargs: Any,
    ) -> "DaskExecutor":
        from distributed import Client

        job_resources = job_resources.copy()
        address = job_resources.pop("address", None)
        if address is None:
            address = os.environ.get("DASK_ADDRESS", None)

        client = Client(address=address)
        return cls(client, job_resources=job_resources)

    @classmethod
    def as_completed(cls, futures: List["Future[_T]"]) -> Iterator["Future[_T]"]:
        from distributed import as_completed

        return as_completed(futures)

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> "Future[_T]":
        if "__cfut_options" in kwargs:
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]

            __fn = cast(
                Callable[_P, _T],
                partial(
                    MultiprocessingExecutor._execute_and_persist_function,
                    output_pickle_path,
                    __fn,
                ),
            )

        __env = os.environ.copy()
        __env["PWD"] = os.getcwd()
        kwargs["__env"] = __env

        # We run the functions in dask as a separate process to not hold the
        # GIL for too long, because dask workers need to be able to communicate
        # with the scheduler regularly.
        __fn = partial(_run_with_nanny, __fn)

        currently_available_workers = self.client.scheduler_info()["workers"]

        def check_resources(
            job_resources: Optional[Dict[str, float]],
            worker_resources: Optional[Dict[str, float]],
        ) -> bool:
            if job_resources is None or len(job_resources) == 0:
                return True
            if worker_resources is None:
                return False
            for key, value in job_resources.items():
                if worker_resources.get(key, 0) < value:
                    return False
            return True

        assert any(
            check_resources(self.job_resources, worker.get("resources"))
            for worker in currently_available_workers.values()
        ), (
            "Requested resources are not available on any currently available worker. "
            + f"Requested resources: {self.job_resources}. Available workers: "
            + str(
                [
                    f"{w['id']} => {w.get('resources', {})}"
                    for w in currently_available_workers.values()
                ]
            )
        )

        fut = self.client.submit(
            partial(__fn, *args, **kwargs), pure=False, resources=self.job_resources
        )

        self.pending_futures.add(fut)
        fut.add_done_callback(self.pending_futures.remove)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def map_unordered(self, fn: Callable[[_S], _T], args: Iterable[_S]) -> Iterator[_T]:
        futs: List["Future[_T]"] = self.map_to_futures(fn, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator() -> Iterator:
            for fut in self.as_completed(futs):
                yield fut.result()

        return result_generator()

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[
            _S
        ],  # TODO change: allow more than one arg per call # noqa FIX002 Line contains TODO
        output_pickle_path_getter: Optional[Callable[[_S], os.PathLike]] = None,
    ) -> List["Future[_T]"]:
        if output_pickle_path_getter is not None:
            futs = [
                self.submit(  # type: ignore[call-arg]
                    fn,
                    arg,
                    __cfut_options={
                        "output_pickle_path": output_pickle_path_getter(arg)
                    },
                )
                for arg in args
            ]
        else:
            futs = [self.submit(fn, arg) for arg in args]

        return futs

    def map(  # type: ignore[override]
        self,
        fn: Callable[[_S], _T],
        iterables: Iterable[_S],
        timeout: Optional[float] = None,
        chunksize: Optional[int] = None,
    ) -> Iterator[_T]:
        if chunksize is None:
            chunksize = 1
        return super().map(fn, iterables, timeout=timeout, chunksize=chunksize)

    def forward_log(self, fut: "Future[_T]") -> _T:
        return fut.result()

    def handle_kill(
        self,
        existing_sigint_handler: Any,
        signum: Optional[int],
        frame: Any,
    ) -> None:
        if self.is_shutting_down:
            return

        self.is_shutting_down = True

        self.client.cancel(list(self.pending_futures))

        if (
            existing_sigint_handler != signal.default_int_handler
            and callable(existing_sigint_handler)  # Could also be signal.SIG_IGN
        ):
            existing_sigint_handler(signum, frame)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:  # noqa: ARG002 Unused method argument: `cancel_futures`
        if wait:
            for fut in list(self.pending_futures):
                fut.result()
            self.client.close(timeout=60 * 60)  # 1 hour
        else:
            self.client.close()

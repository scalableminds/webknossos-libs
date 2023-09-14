import os
from concurrent import futures
from concurrent.futures import Future
from functools import partial
from typing import Any, Callable, Iterable, Iterator, List, Optional, TypeVar, cast

from dask.distributed import Client, as_completed
from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict, MultiprocessingExecutor

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


class DaskExecutor(futures.Executor):
    client: Client

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        self.client = Client(**kwargs)

    @classmethod
    def as_completed(cls, futures: List[Future[_T]]) -> Iterator[Future[_T]]:
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

            __fn = partial(
                MultiprocessingExecutor._execute_and_persist_function,
                output_pickle_path,
                __fn,
            )
        fut = self.client.submit(partial(__fn, *args, **kwargs))

        enrich_future_with_uncaught_warning(fut)
        return fut

    def map_unordered(self, fn: Callable[_P, _T], args: Any) -> Iterator[_T]:
        futs: List["Future[_T]"] = self.map_to_futures(fn, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator() -> Iterator:
            for fut in as_completed(futs):
                yield fut.result()

        return result_generator()

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],  # TODO change: allow more than one arg per call
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

    def map(
        self,
        fn: Callable[[_S], _T],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> Iterator[_T]:
        return iter(
            list(super().map(fn, *iterables, timeout=timeout, chunksize=chunksize))
        )

    def forward_log(self, fut: "Future[_T]") -> _T:
        return fut.result()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if wait:
            self.client.close(timeout=60 * 60 * 24)
        else:
            self.client.close()

import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar, cast

from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict, MultiprocessingExecutor

_T = TypeVar("_T")
_S = TypeVar("_S")
_P = ParamSpec("_P")


class SequentialExecutor(Executor):
    """
    The same as MultiprocessingExecutor, but synchronous and uses only one core.
    """

    def __init__(
        self,
        **__kwargs: Any,
    ) -> None:
        pass

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        fut: Future[_T] = Future()
        if "__cfut_options" in kwargs:
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]
            result = MultiprocessingExecutor._execute_and_persist_function(
                Path(output_pickle_path),
                __fn,
                *args,
                **kwargs,
            )
        else:
            result = __fn(*args, **kwargs)

        fut.set_result(result)
        enrich_future_with_uncaught_warning(fut)
        return fut

    @classmethod
    def as_completed(cls, futures: list[Future[_T]]) -> Iterator[Future[_T]]:
        return as_completed(futures)

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],
        output_pickle_path_getter: Callable[[_S], PathLike] | None = None,
    ) -> list[Future[_T]]:
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
        timeout: float | None = None,
        chunksize: int | None = None,
    ) -> Iterator[_T]:
        if timeout is not None:
            warnings.warn(
                "timeout is not implemented for SequentialExecutor.map",
                category=UserWarning,
            )
        if chunksize is not None:
            warnings.warn(
                "chunksize is not implemented for SequentialExecutor.map",
                category=UserWarning,
            )
        for item in iterables:
            yield fn(item)

    def forward_log(self, fut: Future[_T]) -> _T:
        return fut.result()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass

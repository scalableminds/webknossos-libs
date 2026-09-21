import warnings
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, Future, as_completed
from os import PathLike
from typing import Any, TypeVar

from typing_extensions import ParamSpec

from cluster_tools._utils.cfut_options import (
    cfut_options_kwargs,
    execute_and_persist,
    parse_cfut_options,
)
from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.output_store import FileOutputStore, OutputStore

_T = TypeVar("_T")
_S = TypeVar("_S")
_P = ParamSpec("_P")


class SequentialExecutor(Executor):
    """
    The same as MultiprocessingExecutor, but synchronous and uses only one core.
    """

    def __init__(
        self,
        output_store: OutputStore | None = None,
        **__kwargs: Any,
    ) -> None:
        self.output_store = FileOutputStore() if output_store is None else output_store

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        fut: Future[_T] = Future()
        output_key = parse_cfut_options(kwargs)
        if output_key is not None:
            result = execute_and_persist(
                self.output_store,
                output_key,
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
        output_key_getter: Callable[[_S], str] | None = None,
        output_pickle_path_getter: Callable[[_S], PathLike] | None = None,
    ) -> list[Future[_T]]:
        return [
            self.submit(  # type: ignore[call-arg]
                fn,
                arg,
                **cfut_options_kwargs(
                    arg, output_key_getter, output_pickle_path_getter
                ),
            )
            for arg in args
        ]

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

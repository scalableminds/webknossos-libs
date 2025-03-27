from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from contextlib import AbstractContextManager
from os import PathLike
from typing import (
    Protocol,
    TypeVar,
)

from typing_extensions import ParamSpec

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


class Executor(Protocol, AbstractContextManager["Executor"]):
    @classmethod
    def as_completed(cls, futures: list[Future[_T]]) -> Iterator[Future[_T]]: ...

    def submit(
        self,
        __fn: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]: ...

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],
        output_pickle_path_getter: Callable[[_S], PathLike] | None = None,
    ) -> list[Future[_T]]: ...

    def map(
        self,
        fn: Callable[[_S], _T],
        iterables: Iterable[_S],
        timeout: float | None = None,
        chunksize: int | None = None,
    ) -> Iterator[_T]: ...

    def forward_log(self, fut: Future[_T]) -> _T: ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None: ...

from concurrent.futures import Future
from os import PathLike
from typing import (
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
)

from typing_extensions import ParamSpec

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


class Executor(Protocol, ContextManager["Executor"]):
    @classmethod
    def as_completed(cls, futures: List["Future[_T]"]) -> Iterator["Future[_T]"]: ...

    def submit(
        self,
        __fn: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> "Future[_T]": ...

    def map_unordered(
        self, fn: Callable[[_S], _T], args: Iterable[_S]
    ) -> Iterator[_T]: ...

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],
        output_pickle_path_getter: Optional[Callable[[_S], PathLike]] = None,
    ) -> List["Future[_T]"]: ...

    def map(
        self,
        fn: Callable[[_S], _T],
        iterables: Iterable[_S],
        timeout: Optional[float] = None,
        chunksize: Optional[int] = None,
    ) -> Iterator[_T]: ...

    def forward_log(self, fut: "Future[_T]") -> _T: ...

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None: ...

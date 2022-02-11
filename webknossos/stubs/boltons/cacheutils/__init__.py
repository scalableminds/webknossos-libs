from typing import Any, Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")
KT = TypeVar("KT")


# Taken from https://github.com/wcooley/python-boltons-stubs/blob/master/boltons-stubs/cacheutils.pyi#L127-L130
# Note: The KT type var was added to fix the typing.
class cachedproperty(Generic[T, KT]):
    def __init__(self, func: Callable[[KT], T]) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __get__(self, obj: Any, objtype: Optional[Type]) -> T:
        ...

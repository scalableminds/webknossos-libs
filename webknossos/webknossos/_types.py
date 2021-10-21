from typing import Any

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Openable(Protocol):
    def open(self, mode: str) -> Any:
        ...

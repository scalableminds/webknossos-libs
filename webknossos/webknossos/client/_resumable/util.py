from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import attr


class CallbackDispatcher:
    """Dispatch callbacks to registered targets."""

    def __init__(self) -> None:
        self.targets: list[Callable] = []

    def register(self, callback: Callable) -> None:
        self.targets.append(callback)

    def trigger(self, *args: Any, **kwargs: Any) -> None:
        """Trigger this dispatcher.

        All arguments are passed through to the registered callbacks.
        """
        for callback in self.targets:
            callback(*args, **kwargs)


@attr.frozen
class Config:
    """The configuration for a resumable session."""

    target: str
    chunk_size: int
    simultaneous_uploads: int
    headers: dict[str, Any]
    test_chunks: bool
    max_chunk_retries: int
    permanent_errors: Sequence[int]
    additional_query_params: dict[str, Any]
    generate_unique_identifier: Callable[[Path, Path], str]

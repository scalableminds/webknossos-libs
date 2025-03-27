import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_T = TypeVar("_T")


def warn_after(
    job: str, seconds: int
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Use as decorator to warn when a function is taking longer than {seconds} seconds.
    """

    def outer(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            exceeded_timeout = [False]
            start_time = time.time()

            def warn_function() -> None:
                logging.warning(
                    f"Function {job} is taking suspiciously long (longer than {seconds} seconds)"
                )
                exceeded_timeout[0] = True

            timer = threading.Timer(seconds, warn_function)
            timer.start()

            try:
                result = fn(*args, **kwargs)
                if exceeded_timeout[0]:
                    end_time = time.time()
                    logging.warning(
                        f"Function {job} succeeded after all (took {int(end_time - start_time)} seconds)"
                    )
            finally:
                timer.cancel()
            return result

        return inner

    return outer


def enrich_future_with_uncaught_warning(f: Future) -> None:
    def warn_on_exception(future: Future) -> None:
        maybe_exception = future.exception()
        if maybe_exception is not None:
            logging.error(
                f"A future crashed with an exception: {maybe_exception}. Future: {future}"
            )

    if not hasattr(f, "is_wrapped_by_cluster_tools"):
        f.is_wrapped_by_cluster_tools = True  # type: ignore[attr-defined]
        f.add_done_callback(warn_on_exception)

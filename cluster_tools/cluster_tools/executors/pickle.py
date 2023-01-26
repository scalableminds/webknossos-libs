from concurrent.futures import Future
from typing import Any, Callable, TypeVar

from cluster_tools import pickling

from .multiprocessing import MultiprocessingExecutor

T = TypeVar("T")


def _pickle_identity(obj: T) -> T:
    return pickling.loads(pickling.dumps(obj))


def _pickle_identity_executor(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    result = fn(*args, **kwargs)
    return _pickle_identity(result)


class PickleExecutor(MultiprocessingExecutor):
    """
    The same as MultiprocessingExecutor, but always pickles input and output of the jobs.
    When using this executor for automated tests, it is ensured that using cluster executors in production
    won't provoke pickling-related problems.
    """

    def submit(  # type: ignore[override]
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[T]:
        (fn_pickled, args_pickled, kwargs_pickled) = _pickle_identity((fn, args, kwargs))
        return super().submit(
            _pickle_identity_executor, fn_pickled, *args_pickled, **kwargs_pickled
        )

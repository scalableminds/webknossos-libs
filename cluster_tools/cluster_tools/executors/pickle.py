from concurrent.futures import Future
from typing import Any, Callable, TypeVar

from cluster_tools._utils import pickling
from cluster_tools.executors.multiprocessing import MultiprocessingExecutor

_T = TypeVar("_T")


def _pickle_identity(obj: _T) -> _T:
    return pickling.loads(pickling.dumps(obj))


def _pickle_identity_executor(fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
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
        fn: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> "Future[_T]":
        (fn_pickled, args_pickled, kwargs_pickled) = _pickle_identity(
            (fn, args, kwargs)
        )
        return super().submit(
            _pickle_identity_executor, fn_pickled, *args_pickled, **kwargs_pickled
        )

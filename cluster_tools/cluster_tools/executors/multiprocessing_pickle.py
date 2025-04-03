from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from typing import TypeVar

from typing_extensions import ParamSpec

from cluster_tools._utils import pickling
from cluster_tools.executors.multiprocessing_ import MultiprocessingExecutor

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


def _pickle_identity(obj: _S) -> _S:
    return pickling.loads(pickling.dumps(obj))


def _pickle_identity_executor(
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
    result = fn(*args, **kwargs)
    return _pickle_identity(result)


class MultiprocessingPickleExecutor(MultiprocessingExecutor):
    """
    The same as MultiprocessingExecutor, but always pickles input and output of the jobs.
    When using this executor for automated tests, it is ensured that using cluster executors in production
    won't provoke pickling-related problems.
    """

    def submit(  # type: ignore[override]
        self,
        fn: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        (fn_pickled, args_pickled, kwargs_pickled) = _pickle_identity(
            (fn, args, kwargs)
        )
        return super().submit(
            partial(_pickle_identity_executor, fn_pickled),
            *args_pickled,
            **kwargs_pickled,
        )

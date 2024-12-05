from concurrent.futures import Future
from functools import partial
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from cluster_tools.executors.multiprocessing_pickle import (
    _pickle_identity,
    _pickle_identity_executor,
)
from cluster_tools.executors.sequential import SequentialExecutor

_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


class SequentialPickleExecutor(SequentialExecutor):
    """
    The same as SequentialExecutor, but always pickles input and output of the jobs.
    When using this executor for automated tests, it is ensured that using cluster executors in production
    won't provoke pickling-related problems. In contrast to the MultiprocessingPickleExecutor this executor
    does not have multiprocessing overhead.
    """

    def submit(
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        (fn_pickled, args_pickled, kwargs_pickled) = _pickle_identity(
            (__fn, args, kwargs)
        )
        return super().submit(
            partial(_pickle_identity_executor, fn_pickled),
            *args_pickled,
            **kwargs_pickled,
        )

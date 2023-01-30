from concurrent.futures import Future
from multiprocessing.context import BaseContext
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing import CFutDict, MultiprocessingExecutor
from cluster_tools.executors.sequential import SequentialExecutor

_T = TypeVar("_T")
_P = ParamSpec("_P")


class DebugSequentialExecutor(SequentialExecutor):
    """
    Only use for debugging purposes. This executor does not spawn new processes for its jobs. Therefore,
    setting breakpoint()'s should be possible without context-related problems.
    """

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> "Future[_T]":
        if "__cfut_options" in kwargs:
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]
            fut = self._blocking_submit(
                MultiprocessingExecutor._execute_and_persist_function,  # type: ignore[arg-type]
                output_pickle_path,  # type: ignore[arg-type]
                __fn,  # type: ignore[arg-type]
                *args,
                **kwargs,
            )
        else:
            fut = self._blocking_submit(__fn, *args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def _blocking_submit(
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> "Future[_T]":
        fut: "Future[_T]" = Future()
        result = __fn(*args, **kwargs)
        fut.set_result(result)
        return fut

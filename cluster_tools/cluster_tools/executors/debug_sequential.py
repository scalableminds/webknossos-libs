from concurrent.futures import Future
from pathlib import Path
from typing import Callable, TypeVar, cast

from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict, MultiprocessingExecutor
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
        fut: "Future[_T]" = Future()
        if "__cfut_options" in kwargs:
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]
            result = MultiprocessingExecutor._execute_and_persist_function(
                Path(output_pickle_path),
                __fn,
                *args,
                **kwargs,
            )
        else:
            result = __fn(*args, **kwargs)

        fut.set_result(result)
        enrich_future_with_uncaught_warning(fut)
        return fut

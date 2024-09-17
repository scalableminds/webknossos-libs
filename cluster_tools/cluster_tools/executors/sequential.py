from concurrent.futures import Future
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

from typing_extensions import ParamSpec

from cluster_tools._utils.warning import enrich_future_with_uncaught_warning
from cluster_tools.executors.multiprocessing_ import CFutDict, MultiprocessingExecutor

_T = TypeVar("_T")
_P = ParamSpec("_P")


# Strictly speaking, this executor doesn't need to inherit from MultiprocessingExecutor
# but could inherit from futures.Executor instead. However, this would require to duplicate
# quite a few methods to adhere to the executor protocol (as_completed, map_to_futures, map, forward_log, shutdown).
class SequentialExecutor(MultiprocessingExecutor):
    """
    The same as MultiprocessingExecutor, but synchronous and uses only one core.
    """

    def __init__(
        self,
        *,
        start_method: Optional[str] = None,
        mp_context: Optional[BaseContext] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        **__kwargs: Any,
    ) -> None:
        super().__init__(
            max_workers=1,
            start_method=start_method,
            mp_context=mp_context,
            initializer=initializer,
            initargs=initargs,
        )

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        fut: Future[_T] = Future()
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

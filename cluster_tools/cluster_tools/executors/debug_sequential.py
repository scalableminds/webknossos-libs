from concurrent.futures import Future
from multiprocessing.context import BaseContext
from typing import Any, Callable, Optional, Tuple, TypeVar

from .multiprocessing import CFutDict, MultiprocessingExecutor
from .sequential import SequentialExecutor
from .util import enrich_future_with_uncaught_warning

T = TypeVar("T")


class DebugSequentialExecutor(SequentialExecutor):
    """
    Only use for debugging purposes. This executor does not spawn new processes for its jobs. Therefore,
    setting breakpoint()'s should be possible without context-related problems.
    """

    def submit(  # type: ignore[override]
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        __cfut_options: Optional[CFutDict] = None,
        **kwargs: Any,
    ) -> Future[T]:

        output_pickle_path = None
        if __cfut_options is not None:
            output_pickle_path = kwargs["__cfut_options"]["output_pickle_path"]
            del kwargs["__cfut_options"]

        if output_pickle_path is not None:
            fut = self._blocking_submit(
                MultiprocessingExecutor._execute_and_persist_function,
                output_pickle_path,
                *args,
                **kwargs,
            )
        else:
            fut = self._blocking_submit(*args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def _blocking_submit(
        self,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[T]:
        fut: Future = Future()
        result = fn(*args, **kwargs)
        fut.set_result(result)
        return fut

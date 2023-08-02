from multiprocessing.context import BaseContext
from typing import Any, Callable, Optional, Tuple

from cluster_tools.executors.multiprocessing_ import MultiprocessingExecutor


class SequentialExecutor(MultiprocessingExecutor):
    """
    The same as MultiprocessingExecutor, but always uses only one core. In essence,
    this is a sequential executor approach, but it still makes use of the standard pool approach.
    That way, switching between different executors should always work without any problems.
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

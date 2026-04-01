import math
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, as_completed
from functools import partial
from itertools import islice
from os import PathLike
from types import TracebackType
from typing import TypeVar

from typing_extensions import ParamSpec

from cluster_tools.executor_protocol import Executor

_T = TypeVar("_T")
_S = TypeVar("_S")
_P = ParamSpec("_P")


def _apply_fn_to_batch(fn: Callable, batch: list) -> list:
    return [fn(item) for item in batch]


def _iter_batches(iterable: Iterable[_S], batch_size: int) -> Iterator[list[_S]]:
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch


class BatchingExecutor:
    """
    Wraps another executor and groups items into batches before submission.

    Specify exactly one of:
    - batch_size: fixed number of items per job (must be > 0)
    - target_job_count: desired number of jobs; batch size is computed as
      ceil(n_items / target_job_count) for each map call (must be > 0)
    """

    _executor: Executor
    _batch_size: int | None
    _target_job_count: int | None

    def __init__(
        self,
        executor: Executor,
        *,
        batch_size: int | None = None,
        target_job_count: int | None = None,
    ) -> None:
        if batch_size is not None and target_job_count is not None:
            raise ValueError("Specify either batch_size or target_job_count, not both")
        if batch_size is None and target_job_count is None:
            raise ValueError("Either batch_size or target_job_count must be specified")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if target_job_count is not None and target_job_count <= 0:
            raise ValueError("target_job_count must be greater than 0")
        self._executor = executor
        self._batch_size = batch_size
        self._target_job_count = target_job_count

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def target_job_count(self) -> int | None:
        return self._target_job_count

    def _resolve_batch_size(self, n_items: int) -> int:
        if self._batch_size is not None:
            return self._batch_size
        assert self._target_job_count is not None
        if n_items == 0:
            return 1
        return math.ceil(n_items / self._target_job_count)

    def __enter__(self) -> "BatchingExecutor":
        self._executor.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._executor.__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def as_completed(cls, futures: list[Future[_T]]) -> Iterator[Future[_T]]:
        return as_completed(futures)

    def submit(
        self,
        __fn: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        return self._executor.submit(__fn, *args, **kwargs)

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],
        output_pickle_path_getter: Callable[[_S], PathLike] | None = None,
    ) -> list[Future[_T]]:
        if output_pickle_path_getter is not None:
            raise NotImplementedError(
                "BatchingExecutor does not support output_pickle_path_getter"
            )

        items = list(args)
        batch_size = self._resolve_batch_size(len(items))
        all_item_futures: list[Future[_T]] = []

        for batch in _iter_batches(items, batch_size):
            (batch_future,) = self._executor.map_to_futures(
                partial(_apply_fn_to_batch, fn), [batch]
            )
            item_futures: list[Future[_T]] = [Future() for _ in batch]
            all_item_futures.extend(item_futures)

            def on_batch_done(
                bf: Future[list[_T]], ifs: list[Future[_T]] = item_futures
            ) -> None:
                try:
                    results = bf.result()
                    for f, r in zip(ifs, results):
                        f.set_result(r)
                except Exception as e:
                    for f in ifs:
                        f.set_exception(e)

            batch_future.add_done_callback(on_batch_done)

        return all_item_futures

    def map(
        self,
        fn: Callable[[_S], _T],
        iterables: Iterable[_S],
        timeout: float | None = None,
        chunksize: int | None = None,
    ) -> Iterator[_T]:
        if chunksize is not None:
            raise ValueError(
                "BatchingExecutor does not support chunksize. Use batch_size instead."
            )

        items = list(iterables)
        batch_size = self._resolve_batch_size(len(items))

        def result_generator() -> Iterator[_T]:
            for batch_results in self._executor.map(
                partial(_apply_fn_to_batch, fn),
                _iter_batches(items, batch_size),
                timeout=timeout,
            ):
                yield from batch_results

        return result_generator()

    def forward_log(self, fut: Future[_T]) -> _T:
        return self._executor.forward_log(fut)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

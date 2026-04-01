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
    Each call to the underlying executor processes batch_size items at once.
    """

    batch_size: int
    _executor: Executor

    def __init__(self, executor: Executor, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self._executor = executor
        self.batch_size = batch_size

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

        all_item_futures: list[Future[_T]] = []

        for batch in _iter_batches(args, self.batch_size):
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

        def result_generator() -> Iterator[_T]:
            for batch_results in self._executor.map(
                partial(_apply_fn_to_batch, fn),
                _iter_batches(iterables, self.batch_size),
                timeout=timeout,
            ):
                yield from batch_results

        return result_generator()

    def forward_log(self, fut: Future[_T]) -> _T:
        return self._executor.forward_log(fut)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

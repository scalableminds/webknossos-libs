import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent import futures
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import (
    Any,
    TypedDict,
    TypeVar,
    cast,
)

from typing_extensions import ParamSpec

from cluster_tools._utils import pickling
from cluster_tools._utils.multiprocessing_logging_handler import (
    _MultiprocessingLoggingHandlerPool,
)
from cluster_tools._utils.warning import enrich_future_with_uncaught_warning

# The module name includes a _-suffix to avoid name clashes with the standard library multiprocessing module.


class CFutDict(TypedDict):
    output_pickle_path: str


_T = TypeVar("_T")
_P = ParamSpec("_P")
_S = TypeVar("_S")


class MultiprocessingExecutor(ProcessPoolExecutor):
    """
    Wraps the ProcessPoolExecutor to add various features:
    - map_to_futures method
    - pickling of job's output (see output_pickle_path_getter and output_pickle_path)
    """

    _mp_context: BaseContext

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        start_method: str | None = None,
        mp_context: BaseContext | None = None,
        initializer: Callable | None = None,
        initargs: tuple = (),
        **__kwargs: Any,
    ) -> None:
        if mp_context is None:
            if start_method is not None:
                mp_context = multiprocessing.get_context(start_method)
            elif "MULTIPROCESSING_DEFAULT_START_METHOD" in os.environ:
                mp_context = multiprocessing.get_context(
                    os.environ["MULTIPROCESSING_DEFAULT_START_METHOD"]
                )
            else:
                mp_context = multiprocessing.get_context("spawn")
        else:
            assert start_method is None, (
                "Cannot use both `start_method` and `mp_context` kwargs."
            )

        super().__init__(
            mp_context=mp_context,
            max_workers=max_workers,
            initializer=initializer,
            initargs=initargs,
        )
        if self._mp_context.get_start_method() == "fork":
            self._mp_logging_handler_pool = None
        else:
            self._mp_logging_handler_pool = _MultiprocessingLoggingHandlerPool()

    @classmethod
    def as_completed(cls, futs: list[Future[_T]]) -> Iterator[Future[_T]]:
        return futures.as_completed(futs)

    def submit(  # type: ignore[override]
        self,
        __fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        if "__cfut_options" in kwargs:
            output_pickle_path = cast(CFutDict, kwargs["__cfut_options"])[
                "output_pickle_path"
            ]
            del kwargs["__cfut_options"]
        else:
            output_pickle_path = None

        # Depending on the start_method and output_pickle_path, setup functions may need to be
        # executed in the new process context, before the actual code is ran.
        # These wrapper functions consume their arguments from *args, **kwargs and assume
        # that the next and last argument will be another function that is then called.
        # Eventually, the actually submitted function will be called.

        if output_pickle_path is not None:
            __fn = cast(
                Callable[_P, _T],
                partial(
                    MultiprocessingExecutor._execute_and_persist_function,
                    Path(output_pickle_path),
                    __fn,
                ),
            )

        if self._mp_logging_handler_pool is not None:
            # If a start_method other than the default "fork" is used, logging needs to be re-setup,
            # because the programming context is not inherited in those cases.
            multiprocessing_logging_setup_fn = (
                self._mp_logging_handler_pool.get_multiprocessing_logging_setup_fn()
            )
            __fn = cast(
                Callable[_P, _T],
                partial(
                    MultiprocessingExecutor._setup_logging_and_execute,
                    multiprocessing_logging_setup_fn,
                    __fn,
                ),
            )

        fut = super().submit(__fn, *args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def map(  # type: ignore[override]
        self,
        fn: Callable[[_S], _T],
        iterables: Iterable[_S],
        timeout: float | None = None,
        chunksize: int | None = None,
    ) -> Iterator[_T]:
        if chunksize is None:
            chunksize = 1
        return super().map(fn, iterables, timeout=timeout, chunksize=chunksize)

    @staticmethod
    def _setup_logging_and_execute(
        multiprocessing_logging_setup_fn: Callable[[], None],
        fn: Callable[_P, _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        multiprocessing_logging_setup_fn()
        return fn(*args, **kwargs)

    @staticmethod
    def _execute_and_persist_function(
        output_pickle_path: Path,
        fn: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            logging.warning(f"Job computation failed with:\n{exc.__repr__()}")
            raise exc
        else:
            # Only pickle the result in the success case, since the output
            # is used as a checkpoint.
            # Note that this behavior differs a bit from the cluster executor
            # which will always serialize the output (even exceptions) to
            # disk. However, the output will have a .preliminary prefix at first
            # which is only removed in the success case so that a checkpoint at
            # the desired target only exists if the job was successful.
            with output_pickle_path.open("wb") as file:
                pickling.dump((True, result), file)
            return result

    def map_to_futures(
        self,
        fn: Callable[[_S], _T],
        args: Iterable[_S],
        output_pickle_path_getter: Callable[[_S], os.PathLike] | None = None,
    ) -> list[Future[_T]]:
        if output_pickle_path_getter is not None:
            futs = [
                self.submit(  # type: ignore[call-arg]
                    fn,
                    arg,
                    __cfut_options={
                        "output_pickle_path": output_pickle_path_getter(arg)
                    },
                )
                for arg in args
            ]
        else:
            futs = [self.submit(fn, arg) for arg in args]

        return futs

    def forward_log(self, fut: Future[_T]) -> _T:
        """
        Similar to the cluster executor, this method Takes a future from which the log file is forwarded to the active
        process. This method blocks as long as the future is not done.
        """

        # Since the default behavior of process pool executors is to show the log in the main process
        # we don't need to do anything except for blocking until the future is done.
        return fut.result()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if self._mp_logging_handler_pool is not None:
            self._mp_logging_handler_pool.close()

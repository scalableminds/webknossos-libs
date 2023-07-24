import argparse
import calendar
import functools
import json
import logging
import os
import stat
import sys
import time
import warnings
from concurrent.futures import as_completed
from concurrent.futures._base import Future
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from errno import EACCES, EEXIST, EISDIR
from inspect import getframeinfo, stack
from multiprocessing import cpu_count
from os.path import relpath
from pathlib import Path
from shutil import copyfileobj
from threading import local
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

import numpy as np
import rich
from rich.progress import Progress
from upath import UPath

from cluster_tools import Executor, get_executor

times = {}


def time_start(identifier: str) -> None:
    times[identifier] = time.time()
    logging.debug("{} started".format(identifier))


def time_stop(identifier: str) -> None:
    _time = times.pop(identifier)
    logging.debug("{} took {:.8f}s".format(identifier, time.time() - _time))


def get_executor_for_args(
    args: Optional[argparse.Namespace],
    executor: Optional[Executor] = None,
) -> ContextManager[Executor]:
    if executor is not None:
        return nullcontext(enter_result=executor)

    if args is None:
        # For backwards compatibility with code from other packages
        # we allow args to be None. In this case we are defaulting
        # to these values:
        jobs = cpu_count()
        executor = get_executor("multiprocessing", max_workers=jobs)
        logging.info("Using pool of {} workers.".format(jobs))
    elif args.distribution_strategy == "multiprocessing":
        # Also accept "processes" instead of job to be compatible with segmentation-tools.
        # In the long run, the args should be unified and provided by the clustertools.
        if "jobs" in args:
            jobs = args.jobs
        elif "processes" in args:
            jobs = args.processes
        else:
            jobs = cpu_count()

        executor = get_executor("multiprocessing", max_workers=jobs)
        logging.info("Using pool of {} workers.".format(jobs))
    elif args.distribution_strategy in ("slurm", "kubernetes"):
        if args.job_resources is None:
            resources_example = (
                '{"mem": "1G"}'
                if args.distribution_strategy == "slurm"
                else '{"memory": "1G"}'
            )
            raise argparse.ArgumentTypeError(
                f"Job resources (--job_resources) has to be provided when using {args.distribution_strategy} as distribution strategy. Example: --job_resources='{resources_example}'"
            )

        executor = get_executor(
            args.distribution_strategy,
            debug=True,
            keep_logs=True,
            job_resources=json.loads(args.job_resources),
        )
        logging.info(f"Using {args.distribution_strategy} cluster.")
    elif args.distribution_strategy == "debug_sequential":
        executor = get_executor(
            args.distribution_strategy,
            debug=True,
            keep_logs=True,
        )
    else:
        logging.error(
            "Unknown distribution strategy: {}".format(args.distribution_strategy)
        )

    return executor


F = Callable[..., Any]


def named_partial(func: F, *args: Any, **kwargs: Any) -> F:
    # Propagate __name__ and __doc__ attributes to partial function
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    if hasattr(func, "__annotations__"):
        # Generic types cannot be pickled in Python <= 3.6, see https://github.com/python/typing/issues/511
        partial_func.__annotations__ = {}
    return partial_func


def wait_and_ensure_success(
    futures: List[Future], progress_desc: Optional[str] = None
) -> List[Any]:
    """Waits for all futures to complete and raises an exception
    as soon as a future resolves with an error."""

    results = []
    if progress_desc is None:
        for fut in as_completed(futures):
            results.append(fut.result())
    else:
        with get_rich_progress() as progress:
            task = progress.add_task(progress_desc, total=len(futures))
            for fut in as_completed(futures):
                results.append(fut.result())
                progress.update(task, advance=1)
    return results


def snake_to_camel_case(snake_case_name: str) -> str:
    parts = snake_case_name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def get_chunks(arr: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


def time_since_epoch_in_ms() -> int:
    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    return unixtime * 1000


def copy_directory_with_symlinks(
    src_path: Path,
    dst_path: Path,
    ignore: Iterable[str] = tuple(),
    make_relative: bool = False,
) -> None:
    """
    Links all directories in src_path / dir_name to dst_path / dir_name.
    """
    assert is_fs_path(src_path), f"Cannot create symlink with remote paths {src_path}."
    assert is_fs_path(dst_path), f"Cannot create symlink with remote paths {dst_path}."

    for item in src_path.iterdir():
        if item.name not in ignore:
            symlink_path = dst_path / item.name
            if make_relative:
                rel_or_abspath = Path(relpath(item, symlink_path.parent))
            else:
                rel_or_abspath = item.resolve()
            symlink_path.symlink_to(rel_or_abspath)


def setup_warnings() -> None:
    warnings.filterwarnings("default", category=DeprecationWarning, module="webknossos")


def setup_logging(args: argparse.Namespace) -> None:
    log_path = Path(f"./logs/cuber_{time.strftime('%Y-%m-%d_%H%M%S')}.txt")

    console_log_level = logging.DEBUG if args.verbose else logging.INFO
    file_log_level = logging.DEBUG

    logging_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Always set the global log level to the more verbose of console_log_level and
    # file_log_level to allow to log with different log levels to console and files.
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_log_level, file_log_level))

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_log_level)
    console.setFormatter(logging_formatter)
    root_logger.addHandler(console)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="UTF-8")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(logging_formatter)
    root_logger.addHandler(file_handler)


def add_verbose_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose", help="Verbose output", dest="verbose", action="store_true"
    )

    parser.set_defaults(verbose=False)


def get_rich_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description:<20}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rich.progress.TimeElapsedColumn(),
        "|",
        rich.progress.TimeRemainingColumn(),
    )


def warn_deprecated(deprecated_item: str, alternative_item: str) -> None:
    caller = getframeinfo(stack()[2][0])
    warnings.warn(
        f"[DEPRECATION] `{deprecated_item}` is deprecated, please use `{alternative_item}` instead (see {caller.filename}:{caller.lineno})",
        DeprecationWarning,
        stacklevel=2,
    )


def is_fs_path(path: Path) -> bool:
    return not isinstance(path, UPath)


def strip_trailing_slash(path: Path) -> Path:
    if isinstance(path, UPath):
        return UPath(
            str(path).rstrip("/"),
            **path._kwargs.copy(),
        )
    else:
        return Path(str(path).rstrip("/"))


def rmtree(path: Path) -> None:
    def _walk(path: Path) -> Iterator[Path]:
        if path.exists():
            if path.is_dir() and not path.is_symlink():
                for p in path.iterdir():
                    yield from _walk(p)
            yield path

    for sub_path in _walk(path):
        try:
            if sub_path.is_file() or sub_path.is_symlink():
                sub_path.unlink()
            elif sub_path.is_dir():
                sub_path.rmdir()
        except FileNotFoundError:
            # Some implementations `UPath` do not have explicit directory representations
            # Therefore, directories only exist, if they have files. Consequently, when
            # all files have been deleted, the directory does not exist anymore.
            pass


def copytree(in_path: Path, out_path: Path) -> None:
    def _walk(path: Path, base_path: Path) -> Iterator[Tuple[Path, Path]]:
        yield (path, path.relative_to(base_path))
        if path.is_dir():
            for p in path.iterdir():
                yield from _walk(p, base_path)

    for in_sub_path, sub_path in _walk(in_path, in_path):
        if in_sub_path.is_dir():
            (out_path / sub_path).mkdir(parents=True, exist_ok=True)
        else:
            with (in_path / sub_path).open("rb") as in_file, (out_path / sub_path).open(
                "wb"
            ) as out_file:
                copyfileobj(in_file, out_file)


K = TypeVar("K")  # key
V = TypeVar("V")  # value
C = TypeVar("C")  # cache


class LazyReadOnlyDict(Mapping[K, V]):
    def __init__(self, entries: Dict[K, C], func: Callable[[C], V]) -> None:
        self.entries = entries
        self.func = func

    def __getitem__(self, key: K) -> V:
        return self.func(self.entries[key])

    def __iter__(self) -> Iterator[K]:
        for key in self.entries:
            yield key

    def __len__(self) -> int:
        return len(self.entries)


class NDArrayLike(Protocol):
    def __getitem__(self, selection: Tuple[slice, ...]) -> np.ndarray:
        ...

    def __setitem__(self, selection: Tuple[slice, ...], value: np.ndarray) -> None:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def ndim(self) -> int:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...


@dataclass
class FileLockContext:
    """A dataclass which holds the context for a ``BaseFileLock`` object."""

    # The context is held in a separate class to allow optional use of thread local storage via the
    # ThreadLocalFileContext class.

    #: The path to the lock file.
    lock_file: Path

    #: The default timeout value.
    timeout: float

    #: The mode for the lock files
    mode: int

    #: The file descriptor for the *_lock_file* as it is returned by the Path.open() function, not None when lock held
    lock_file_fd: Any | None = None

    #: The lock counter is used for implementing the nested locking mechanism.
    lock_counter: int = 0  # When the lock is acquired is increased and the lock is only released, when this value is 0


class ThreadLocalFileContext(FileLockContext, local):
    """A thread local version of the ``FileLockContext`` class."""


class Timeout(TimeoutError):  # noqa: N818
    """Raised when the lock could not be acquired in *timeout* seconds."""

    def __init__(self, lock_file: str) -> None:
        super().__init__()
        self._lock_file = lock_file

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, (self._lock_file,)  # Properly pickle the exception

    def __str__(self) -> str:
        return f"The file lock '{self._lock_file}' could not be acquired."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.lock_file!r})"

    @property
    def lock_file(self) -> str:
        """:return: The path of the file lock."""
        return self._lock_file


def raise_on_not_writable_file(filename: Path) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.
    This is done so files that will never be writable can be separated from
    files that are writable but currently locked
    :param filename: file to check
    :raises OSError: as if the file was opened for writing.
    """
    try:  # use stat to do exists + can write to check without race condition
        file_stat = os.stat(filename)  # noqa: PTH116
    except OSError:
        return  # swallow does not exist or other errors

    if (
        file_stat.st_mtime != 0
    ):  # if os.stat returns but modification is zero that's an invalid os.stat - ignore it
        if not (file_stat.st_mode & stat.S_IWUSR):
            raise PermissionError(EACCES, "Permission denied", filename)

        if stat.S_ISDIR(file_stat.st_mode):
            if sys.platform == "win32":  # pragma: win32 cover
                # On Windows, this is PermissionError
                raise PermissionError(EACCES, "Permission denied", filename)
            else:  # pragma: win32 no cover # noqa: RET506
                # On linux / macOS, this is IsADirectoryError
                raise IsADirectoryError(EISDIR, "Is a directory", filename)


class SoftFileLock:
    """A file lock that simply watches the existence of the lock file. It is inspired by: https://github.com/tox-dev/py-filelock"""

    def __init__(
        self,
        lock_file: Path,
        timeout: float = -1,
        mode: int = 0o644,
        thread_local: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """
        Create a new lock object.

        :param lock_file: path to the file
        :param timeout: default timeout when acquiring the lock, in seconds. It will be used as fallback value in
        the acquire method, if no timeout value (``None``) is given. If you want to disable the timeout, set it
        to a negative value. A timeout of 0 means, that there is exactly one attempt to acquire the file lock.
        :param mode: file permissions for the lockfile.
        :param thread_local: Whether this object's internal context should be thread local or not.
        If this is set to ``False`` then the lock will be reentrant across threads.
        """
        self._is_thread_local = thread_local

        # Create the context. Note that external code should not work with the context directly  and should instead use
        # properties of this class.
        kwargs: dict[str, Any] = {
            "lock_file": lock_file,
            "timeout": timeout,
            "mode": mode,
        }
        self._context: FileLockContext = (
            ThreadLocalFileContext if thread_local else FileLockContext
        )(**kwargs)

    def is_thread_local(self) -> bool:
        """:return: a flag indicating if this lock is thread local or not"""
        return self._is_thread_local

    @property
    def lock_file(self) -> Path:
        """:return: path to the lock file"""
        return self._context.lock_file

    @property
    def timeout(self) -> float:
        """
        :return: the default timeout value, in seconds

        .. versionadded:: 2.0.0
        """
        return self._context.timeout

    @timeout.setter
    def timeout(self, value: float | str) -> None:
        """
        Change the default timeout value.

        :param value: the new value, in seconds
        """
        self._context.timeout = float(value)

    @property
    def is_locked(self) -> bool:
        """

        :return: A boolean indicating if the lock file is holding the lock currently.

        .. versionchanged:: 2.0.0

            This was previously a method and is now a property.
        """
        return self._context.lock_file_fd is not None

    @property
    def lock_counter(self) -> int:
        """:return: The number of times this lock has been acquired (but not yet released)."""
        return self._context.lock_counter

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        poll_intervall: float | None = None,
        blocking: bool = True,
    ) -> "SoftFileLock":
        """
        Try to acquire the file lock.

        :param timeout: maximum wait time for acquiring the lock, ``None`` means use the default :attr:`~timeout` is and
         if ``timeout < 0``, there is no timeout and this method will block until the lock could be acquired
        :param poll_interval: interval of trying to acquire the lock file
        :param poll_intervall: deprecated, kept for backwards compatibility, use ``poll_interval`` instead
        :param blocking: defaults to True. If False, function will return immediately if it cannot obtain a lock on the
         first attempt. Otherwise, this method will block until the timeout expires or the lock is acquired.
        :raises Timeout: if fails to acquire lock within the timeout period
        :return: a context object that will unlock the file when the context is exited

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            with lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            lock.acquire()
            try:
                pass
            finally:
                lock.release()

        .. versionchanged:: 2.0.0

            This method returns now a *proxy* object instead of *self*,
            so that it can be used in a with statement without side effects.

        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self._context.timeout

        if poll_intervall is not None:
            msg = "use poll_interval instead of poll_intervall"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            poll_interval = poll_intervall

        # Increment the number right at the beginning. We can still undo it, if something fails.
        self._context.lock_counter += 1

        lock_id = id(self)
        lock_filename = str(self.lock_file)
        start_time = time.perf_counter()
        try:
            while True:
                if not self.is_locked:
                    logging.debug(
                        "Attempting to acquire lock %s on %s", lock_id, lock_filename
                    )
                    self._acquire()
                if self.is_locked:
                    logging.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    break
                if blocking is False:
                    logging.debug(
                        "Failed to immediately acquire lock %s on %s",
                        lock_id,
                        lock_filename,
                    )
                    raise Timeout(lock_filename)  # noqa: TRY301
                if 0 <= timeout < time.perf_counter() - start_time:
                    logging.debug(
                        "Timeout on acquiring lock %s on %s", lock_id, lock_filename
                    )
                    raise Timeout(lock_filename)  # noqa: TRY301
                msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                logging.debug(msg, lock_id, lock_filename, poll_interval)
                time.sleep(poll_interval)
        except BaseException:  # Something did go wrong, so decrement the counter.
            self._context.lock_counter = max(0, self._context.lock_counter - 1)
            raise
        return self

    def release(self, force: bool = False) -> None:  # noqa: FBT001, FBT002
        """
        Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0. Also
        note, that the lock file itself is not automatically deleted.

        :param force: If true, the lock counter is ignored and the lock is released in every case/
        """
        if self.is_locked:
            self._context.lock_counter -= 1

            if self._context.lock_counter == 0 or force:
                lock_id, lock_filename = id(self), self.lock_file

                logging.debug(
                    "Attempting to release lock %s on %s", lock_id, lock_filename
                )
                self._release()
                self._context.lock_counter = 0
                logging.debug("Lock %s released on %s", lock_id, lock_filename)

    def _acquire(self) -> None:
        raise_on_not_writable_file(self.lock_file)
        # first check for exists and read-only mode as the open will mask this case as EEXIST
        try:
            file_handler = self.lock_file.open("x")
        except OSError as exception:  # re-raise unless expected exception
            if not (
                exception.errno == EEXIST  # lock already exist
                or (
                    exception.errno == EACCES and sys.platform == "win32"
                )  # has no access to this lock
            ):  # pragma: win32 no cover
                raise
        else:
            self._context.lock_file_fd = file_handler

    def _release(self) -> None:
        assert self._context.lock_file_fd is not None  # noqa: S101
        self._context.lock_file_fd.close()  # the lock file is definitely not None
        self._context.lock_file_fd = None
        with suppress(OSError):  # the file is already deleted and that's what we want
            Path(self.lock_file).unlink()

    def __enter__(self) -> "SoftFileLock":
        """
        Acquire the lock.

        :return: the lock object
        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release the lock.

        :param exc_type: the exception type if raised
        :param exc_value: the exception value if raised
        :param traceback: the exception traceback if raised
        """
        self.release()

    def __del__(self) -> None:
        """Called when the lock object is deleted."""
        self.release(force=True)

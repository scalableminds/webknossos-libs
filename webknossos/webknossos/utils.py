import argparse
import calendar
import functools
import json
import logging
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from inspect import getframeinfo, stack
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path, PosixPath, WindowsPath
from shutil import copyfileobj, move
from threading import Thread
from typing import (
    Any,
    Protocol,
    TypeVar,
)
from urllib.parse import urlparse

import httpx
import numpy as np
import rich
from cluster_tools import Executor, get_executor
from packaging.version import InvalidVersion, Version
from rich.progress import Progress
from upath import UPath

times = {}


def time_start(identifier: str) -> None:
    times[identifier] = time.time()
    logging.debug(f"{identifier} started")


def time_stop(identifier: str) -> None:
    _time = times.pop(identifier)
    logging.debug(f"{identifier} took {time.time() - _time:.8f}s")


def get_executor_for_args(
    args: argparse.Namespace | None,
    executor: Executor | None = None,
) -> AbstractContextManager[Executor]:
    if executor is not None:
        return nullcontext(enter_result=executor)

    if args is None:
        # For backwards compatibility with code from other packages
        # we allow args to be None. In this case we are defaulting
        # to these values:
        jobs = cpu_count()
        executor = get_executor("multiprocessing", max_workers=jobs)
        logging.info(f"Using pool of {jobs} workers.")
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
        logging.info(f"Using pool of {jobs} workers.")
    elif args.distribution_strategy in ("slurm", "kubernetes"):
        if args.job_resources is None:
            resources_example = (
                '{"mem": "1G"}'
                if args.distribution_strategy == "slurm"
                else '{"memory": "1G"}'
            )
            raise argparse.ArgumentTypeError(
                f"Job resources (--job-resources) has to be provided when using {args.distribution_strategy} as distribution strategy. Example: --job-resources='{resources_example}'"
            )

        executor = get_executor(
            args.distribution_strategy,
            debug=True,
            keep_logs=True,
            job_resources=json.loads(args.job_resources),
        )
        logging.info(f"Using {args.distribution_strategy} cluster.")
    elif args.distribution_strategy == "sequential":
        executor = get_executor(
            args.distribution_strategy,
            debug=True,
            keep_logs=True,
        )
    else:
        logging.error(f"Unknown distribution strategy: {args.distribution_strategy}")

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


def infer_metadata_type(value: str | int | float | Sequence[str]) -> str:
    if isinstance(value, str):
        return "string"
    if isinstance(value, Sequence):
        for i in value:
            if not isinstance(i, str):
                raise ValueError(
                    f"In lists only str type is allowed, got: {type(value)}"
                )
        if all(isinstance(i, str) for i in value):
            return "string[]"
        raise ValueError(f"Unsupported metadata type: {type(value)}")
    if isinstance(value, int | float):
        return "number"
    raise ValueError(f"Unsupported metadata type: {type(value)}")


def parse_metadata_value(
    value: str | list[str], ts_type: str
) -> str | int | float | Sequence[str]:
    if ts_type == "string[]":
        return list(str(v) for v in value)
    elif ts_type == "number":
        assert not isinstance(value, list)
        try:
            return int(value)
        except ValueError:
            return float(value)
    elif ts_type == "string":
        return str(value)
    else:
        raise ValueError(f"Unknown metadata type {ts_type}")


def wait_and_ensure_success(
    futures: list[Future],
    executor: Executor,
    progress_desc: str | None = None,
) -> list[Any]:
    """Waits for all futures to complete and raises an exception
    as soon as a future resolves with an error."""

    results = []
    if progress_desc is None:
        for fut in executor.as_completed(futures):
            results.append(fut.result())  #  noqa: PERF401 Use a list comprehension to create a transformed list
    else:
        with get_rich_progress() as progress:
            task = progress.add_task(progress_desc, total=len(futures))
            for fut in executor.as_completed(futures):
                results.append(fut.result())
                progress.update(task, advance=1)
    return results


def snake_to_camel_case(snake_case_name: str) -> str:
    parts = snake_case_name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def get_chunks(arr: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


def time_since_epoch_in_ms() -> int:
    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    return unixtime * 1000


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


def count_defined_values(values: Iterable[Any | None]) -> int:
    return sum(i is not None for i in values)


def is_fs_path(path: Path) -> bool:
    from upath.implementations.local import PosixUPath, WindowsUPath

    return not isinstance(path, UPath) or isinstance(
        path, PosixPath | WindowsPath | PosixUPath | WindowsUPath
    )


def is_remote_path(path: Path) -> bool:
    return not is_fs_path(path)


def resolve_if_fs_path(path: Path) -> Path:
    if is_fs_path(path):
        return path.resolve()
    return path


def is_writable_path(path: Path) -> bool:
    from upath.implementations.http import HTTPPath

    # cannot write to http paths
    return not isinstance(path, HTTPPath)


def strip_trailing_slash(path: Path) -> Path:
    if isinstance(path, UPath) and not is_fs_path(path):
        return UPath(str(path).rstrip("/"), **path.storage_options)
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
        except FileNotFoundError:  # noqa:  PERF203 `try`-`except` within a loop incurs performance overhead
            # Some implementations `UPath` do not have explicit directory representations
            # Therefore, directories only exist, if they have files. Consequently, when
            # all files have been deleted, the directory does not exist anymore.
            pass


def copytree(
    in_path: Path,
    out_path: Path,
    *,
    threads: int | None = 10,
    progress_desc: str | None = None,
) -> None:
    def _walk(path: Path, base_path: Path) -> Iterator[tuple[Path, tuple[str, ...]]]:
        # base_path.parts is a prefix of path.parts; strip it
        assert len(path.parts) >= len(base_path.parts)
        assert path.parts[: len(base_path.parts)] == base_path.parts
        yield (path, path.parts[len(base_path.parts) :])

        if path.is_dir():
            for p in path.iterdir():
                yield from _walk(p, base_path)

    def _append(path: Path, parts: tuple[str, ...]) -> Path:
        for p in parts:
            path = path / p
        return path

    def _copy(args: tuple[Path, Path, tuple[str, ...]]) -> None:
        in_path, out_path, sub_path = args
        with (
            _append(in_path, sub_path).open("rb") as in_file,
            _append(out_path, sub_path).open("wb") as out_file,
        ):
            copyfileobj(in_file, out_file)

    files_to_copy: list[tuple[Path, Path, tuple[str, ...]]] = []
    for in_sub_path, sub_path in _walk(in_path, in_path):
        if in_sub_path.is_dir():
            _append(out_path, sub_path).mkdir(parents=True, exist_ok=True)
        else:
            files_to_copy.append((in_path, out_path, sub_path))

    with ThreadPool(threads) as pool:
        iterator = pool.imap_unordered(_copy, files_to_copy)

        if progress_desc:
            with get_rich_progress() as progress:
                task = progress.add_task(progress_desc, total=len(files_to_copy))
                for _ in iterator:
                    progress.update(task, advance=1)
        for _ in iterator:
            pass


def movetree(in_path: Path, out_path: Path) -> None:
    move(in_path, out_path)


class LazyPath:
    paths: tuple[Path, ...]
    resolution: int | None = None

    def __init__(self, *paths: Path):
        self.paths = tuple(paths)

    @classmethod
    def resolved(cls, path: Path) -> "LazyPath":
        obj = cls(path)
        obj.resolution = 0
        return obj

    def resolve(self) -> Path:
        if self.resolution is not None:
            return self.paths[self.resolution]
        else:
            for i, path in enumerate(self.paths):
                if path.exists():
                    self.resolution = i
                    return path
            raise FileNotFoundError(f"No path exists in {self.paths}")

    def __repr__(self) -> str:
        if self.resolution is not None:
            return repr(self.paths[self.resolution])
        return f"LazyPath({','.join(repr(p) for p in self.paths)})"

    def __str__(self) -> str:
        if self.resolution is not None:
            return str(self.paths[self.resolution])
        return f"LazyPath({','.join(str(p) for p in self.paths)})"


K = TypeVar("K")  # key
V = TypeVar("V")  # value
C = TypeVar("C")  # cache


class LazyReadOnlyDict(Mapping[K, V]):
    def __init__(self, entries: dict[K, C], func: Callable[[C], V]) -> None:
        self.entries = entries
        self.func = func

    def __getitem__(self, key: K) -> V:
        return self.func(self.entries[key])

    def __iter__(self) -> Iterator[K]:
        yield from self.entries

    def __len__(self) -> int:
        return len(self.entries)


class NDArrayLike(Protocol):
    def __getitem__(self, selection: tuple[slice, ...]) -> np.ndarray: ...

    def __setitem__(self, selection: tuple[slice, ...], value: np.ndarray) -> None: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype: ...


def get_latest_version_from_pypi() -> Version:
    """Get the latest version of the Webknossos package from PyPI."""

    try:
        response = httpx.get("https://pypi.org/pypi/webknossos/json")
        response.raise_for_status()
        data = response.json()
        releases = data["releases"]
        versions = [Version(v) for v in releases.keys()]
        latest_version = max(versions)
        return latest_version

    except (httpx.HTTPError, InvalidVersion):
        # Failed to get latest version from PyPI
        pass

    return Version("0.0.0")


def check_version_in_background(current_version: str) -> None:
    """
    Schedule a non-blocking version check that will log a warning if the current version is outdated.
    This function runs the check in a separate thread to avoid blocking the main application.
    """

    def check_version_in_thread() -> None:
        try:
            # Get the latest version
            latest_version = get_latest_version_from_pypi()

            # Compare versions and log warning if needed
            if Version(current_version) < latest_version:
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Your current version {current_version} of the webknossos-libs is outdated. "
                    f"The latest version available on PyPI is {latest_version}. "
                    f"Consider upgrading to the latest version to avoid being out-of-sync with "
                    f"the latest WEBKNOSSOS features. See GitHub for full changelog of all "
                    f"releases (https://github.com/scalableminds/webknossos-libs/releases)."
                )
        except Exception:
            # Silently ignore any errors in version checking
            pass

    # Start the check in a daemon thread so it won't block program exit
    t = Thread(target=check_version_in_thread, daemon=True)
    t.start()


def safe_is_relative_to(path: Path, base_path: Path) -> bool:
    if (
        (is_fs_path(path) and is_fs_path(base_path))
        or UPath(path).protocol == UPath(base_path).protocol
    ) and path.is_relative_to(base_path):
        return True
    return False


def enrich_path(path: str, dataset_path: Path) -> Path:
    upath = UPath(path)
    if upath.protocol in ("http", "https"):
        from .client.context import _get_context
        from .dataset.defaults import SSL_CONTEXT

        # To setup the mag for non-public remote paths, we need to get the token from the context
        wk_context = _get_context()
        token = wk_context.datastore_token
        return UPath(
            path,
            headers={} if token is None else {"X-Auth-Token": token},
            ssl=SSL_CONTEXT,
        )

    elif upath.protocol == "s3":
        parsed_url = urlparse(str(upath))
        endpoint_url = f"https://{parsed_url.netloc}"
        bucket, key = parsed_url.path.lstrip("/").split("/", maxsplit=1)

        return UPath(
            f"s3://{bucket}/{key}", client_kwargs={"endpoint_url": endpoint_url}
        )

    if not upath.is_absolute():
        return resolve_if_fs_path(dataset_path / upath)
    return resolve_if_fs_path(upath)


def dump_path(path: Path, dataset_path: Path | None) -> str:
    if is_fs_path(path) and not path.is_absolute():
        if dataset_path is None:
            raise ValueError("dataset_path must be provided when path is not absolute.")
        path = dataset_path / path
    path = resolve_if_fs_path(path)
    if dataset_path is not None:
        if str(path).startswith(str(dataset_path)):
            return str(path).removeprefix(str(dataset_path)).lstrip("/")
        if safe_is_relative_to(path, dataset_path):
            return str(path.relative_to(dataset_path))
    if isinstance(path, UPath) and path.protocol == "s3":
        return f"s3://{urlparse(path.storage_options['client_kwargs']['endpoint_url']).netloc}/{path.path}"
    return str(path)

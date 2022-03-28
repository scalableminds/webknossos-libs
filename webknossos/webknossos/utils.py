import argparse
import calendar
import functools
import json
import logging
import sys
import time
import warnings
from concurrent.futures import as_completed
from concurrent.futures._base import Future
from datetime import datetime
from inspect import getframeinfo, stack
from multiprocessing import cpu_count
from os import PathLike
from os.path import relpath
from pathlib import Path
from shutil import copyfileobj
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Union

import rich
from cluster_tools import WrappedProcessPoolExecutor, get_executor
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from rich.progress import Progress
from upath import UPath

times = {}


def time_start(identifier: str) -> None:
    times[identifier] = time.time()
    logging.debug("{} started".format(identifier))


def time_stop(identifier: str) -> None:
    _time = times.pop(identifier)
    logging.debug("{} took {:.8f}s".format(identifier, time.time() - _time))


def get_executor_for_args(
    args: Optional[argparse.Namespace],
) -> Union[ClusterExecutor, WrappedProcessPoolExecutor]:
    executor = None
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
        "[progress.description]{task.description}",
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
    )


def make_upath(maybe_path: Union[str, PathLike, Path]) -> UPath:
    return maybe_path if isinstance(maybe_path, UPath) else UPath(maybe_path)


def is_fs_path(path: Path) -> bool:
    # Distinguish between `pathlib.*Path` and `UPath` through a `UPath`-specific attribute
    return not hasattr(path, "_url")


def is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except NotImplementedError:
        # `Path` raises `NotImplmentedError` for some methods, including `is_symlink`
        return False


def rmtree(path: Path) -> None:
    def _walk(path: Path) -> Iterator[Path]:
        if path.exists():
            if path.is_dir() and not is_symlink(path):
                for p in path.iterdir():
                    yield from _walk(p)
            yield path

    for sub_path in _walk(path):
        try:
            if sub_path.is_file() or is_symlink(sub_path):
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

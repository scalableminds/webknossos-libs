import argparse
import calendar
import functools
import json
import logging
import os
import time
from concurrent.futures import as_completed
from concurrent.futures._base import Future
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

import rich
from cluster_tools import WrappedProcessPoolExecutor, get_executor
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from rich.progress import Progress

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
    elif args.distribution_strategy == "slurm":
        if args.job_resources is None:
            raise argparse.ArgumentTypeError(
                'Job resources (--job_resources) has to be provided when using slurm as distribution strategy. Example: --job_resources=\'{"mem": "10M"}\''
            )

        executor = get_executor(
            "slurm",
            debug=True,
            keep_logs=True,
            job_resources=json.loads(args.job_resources),
        )
        logging.info("Using slurm cluster.")
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
    for item in src_path.iterdir():
        if item.name not in ignore:
            symlink_path = dst_path / item.name
            if make_relative:
                rel_or_abspath = Path(os.path.relpath(item, symlink_path.parent))
            else:
                rel_or_abspath = item.resolve()
            symlink_path.symlink_to(rel_or_abspath)


def setup_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


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

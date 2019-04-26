import time
import wkw
import numpy as np
import logging
import argparse
import cluster_tools
import json
from glob import iglob
from collections import namedtuple
from multiprocessing import cpu_count, Lock
import concurrent
from concurrent.futures import ProcessPoolExecutor
from os import path
from platform import python_version
from math import floor, ceil

from .knossos import KnossosDataset, CUBE_EDGE_LEN

WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "dtype", "mag")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
FallbackArgs = namedtuple("FallbackArgs", ("distribution_strategy", "jobs"))


def open_wkw(info, **kwargs):
    if info.dtype is not None:
        header = wkw.Header(np.dtype(info.dtype), **kwargs)
    else:
        logging.warn(
            "Discarding the following wkw header args, because dtype was not provided: {}".format(
                kwargs
            )
        )
        header = None
    ds = wkw.Dataset.open(
        path.join(info.dataset_path, info.layer_name, str(info.mag)), header
    )
    return ds


def ensure_wkw(target_wkw_info, **kwargs):
    # Open will create the dataset if it doesn't exist yet
    target_wkw = open_wkw(target_wkw_info, **kwargs)
    target_wkw.close()


def open_knossos(info):
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def add_verbose_flag(parser):
    parser.add_argument(
        "--verbose", "-v", help="Verbose output", dest="verbose", action="store_true"
    )

    parser.set_defaults(verbose=False)


def find_files(source_path, extensions):
    # Find all files with a matching file extension
    return (
        f
        for f in iglob(source_path, recursive=True)
        if any([f.endswith(suffix) for suffix in extensions])
    )


def get_chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


# min_z and max_z are both inclusive
def get_regular_chunks(min_z, max_z, chunk_size):
    i = floor(min_z / chunk_size) * chunk_size
    while i < ceil((max_z + 1) / chunk_size) * chunk_size:
        yield range(i, i + chunk_size)
        i += chunk_size


def add_distribution_flags(parser):
    parser.add_argument(
        "--jobs",
        "-j",
        default=cpu_count(),
        type=int,
        help="Number of processes to be spawned.",
    )

    parser.add_argument(
        "--distribution_strategy",
        default="multiprocessing",
        choices=["slurm", "multiprocessing"],
        help="Strategy to distribute the task across CPUs or nodes.",
    )

    parser.add_argument(
        "--job_resources",
        default=None,
        help='Necessary when using slurm as distribution strategy. Should be a JSON string (e.g., --job_resources=\'{"mem": "10M"}\')',
    )


def get_executor_for_args(args):
    if args is None:
        # For backwards compatibility with code from other packages
        # we allow args to be None. In this case we are defaulting
        # to these values:
        args = FallbackArgs("multiprocessing", cpu_count())

    executor = None

    if args.distribution_strategy == "multiprocessing":
        # Also accept "processes" instead of job to be compatible with segmentation-tools.
        # In the long run, the args should be unified and provided by the clustertools.
        if "jobs" in args:
            jobs = args.jobs
        elif "processes" in args:
            jobs = args.processes
        else:
            jobs = cpu_count()

        executor = cluster_tools.get_executor("multiprocessing", jobs)
        logging.info("Using pool of {} workers.".format(jobs))
    elif args.distribution_strategy == "slurm":
        if args.job_resources is None:
            raise argparse.ArgumentTypeError(
                'Job resources (--job_resources) has to be provided when using slurm as distribution strategy. Example: --job_resources=\'{"mem": "10M"}\''
            )

        executor = cluster_tools.get_executor(
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


times = {}


def time_start(identifier):
    times[identifier] = time.time()


def time_stop(identifier):
    _time = times.pop(identifier)
    logging.debug("{} took {:.8f}s".format(identifier, time.time() - _time))


# Waits for all futures to complete and raises an exception
# as soon as a future resolves with an error.
def wait_and_ensure_success(futures):
    for fut in concurrent.futures.as_completed(futures):
        fut.result()

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
from os import path, getpid
from platform import python_version
from math import floor, ceil
from .mag import Mag

from .knossos import KnossosDataset, CUBE_EDGE_LEN

WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "dtype", "mag")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
FallbackArgs = namedtuple("FallbackArgs", ("distribution_strategy", "jobs"))


BLOCK_LEN = 32


def open_wkw(info, **kwargs):
    if hasattr(info, "dtype"):
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
        "--silent", help="Silent output", dest="verbose", action="store_false"
    )

    parser.set_defaults(verbose=True)


def add_scale_flag(parser):
    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25). This is the size of one voxel in nm.",
        required=True,
    )


def add_isotropic_flag(parser):
    parser.add_argument(
        "--isotropic",
        help="Activates isotropic downsampling. The default is anisotropic downsampling. "
        "Isotropic downsampling will always downsample each dimension with the factor 2.",
        dest="isotropic",
        default=False,
        action="store_true",
    )

    parser.set_defaults(anisotropic=False)


def setup_logging(args):

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


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


def add_batch_size_flag(parser):
    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of sections to buffer per job",
        type=int,
        default=BLOCK_LEN,
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

        executor = cluster_tools.get_executor("multiprocessing", max_workers=jobs)
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


class BufferedSliceWriter(object):
    def __init__(
        self,
        dataset_path,
        layer_name,
        dtype,
        bounding_box,
        origin,
        buffer_size=32,
        mag=Mag(1),
    ):

        self.dataset_path = dataset_path
        self.layer_name = layer_name
        self.buffer_size = buffer_size

        layer_path = path.join(self.dataset_path, self.layer_name, mag.to_layer_name())

        self.dataset = wkw.Dataset.open(layer_path, wkw.Header(dtype))
        self.origin = origin
        self.bounding_box = bounding_box

        self.buffer = []
        self.current_z = None
        self.buffer_start_z = None

    def write_slice(self, z: int, data: np.ndarray):
        """Takes in a slice in [y, x] shape, writes to WKW file."""

        if len(self.buffer) == 0:
            self.current_z = z
            self.buffer_start_z = z

        assert (
            z == self.current_z
        ), "({}) Slices have to be written sequentially!".format(getpid())

        self.buffer.append(data.transpose())
        self.current_z += 1

        if self.current_z % self.buffer_size == 0:
            self._write_buffer()

    def _write_buffer(self):

        if len(self.buffer) == 0:
            return

        assert len(self.buffer) <= self.buffer_size

        logging.debug(
            "({}) Writing {} slices at position {}.".format(
                getpid(), len(self.buffer), self.buffer_start_z
            )
        )

        origin_with_offset = self.origin.copy()
        origin_with_offset[2] = self.buffer_start_z
        x_max = max(slice.shape[0] for slice in self.buffer)
        y_max = max(slice.shape[1] for slice in self.buffer)
        self.buffer = [
            np.pad(
                slice,
                mode="constant",
                pad_width=[(0, x_max - slice.shape[0]), (0, y_max - slice.shape[1])],
            )
            for slice in self.buffer
        ]
        data = np.concatenate(
            [np.expand_dims(slice, 2) for slice in self.buffer], axis=2
        )

        self.dataset.write(origin_with_offset, data)

        self.buffer = []

    def close(self):

        self._write_buffer()
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

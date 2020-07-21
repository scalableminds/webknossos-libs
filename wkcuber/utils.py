import re
import time
import wkw
import numpy as np
import logging
import argparse
import cluster_tools
import json
import os
import psutil
from typing import List, Tuple, Union
from glob import iglob
from collections import namedtuple
from multiprocessing import cpu_count
import concurrent
from os import path, getpid
from math import floor, ceil
from logging import getLogger
import traceback
from wkcuber.api.bounding_box import BoundingBox

from .knossos import KnossosDataset
from .mag import Mag

WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "mag", "header")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
FallbackArgs = namedtuple("FallbackArgs", ("distribution_strategy", "jobs"))


BLOCK_LEN = 32
DEFAULT_WKW_FILE_LEN = 32
DEFAULT_WKW_VOXELS_PER_BLOCK = 32
CUBE_REGEX = re.compile(r"z(\d+)/y(\d+)/x(\d+)(\.wkw)$")

logger = getLogger(__name__)


def open_wkw(info):
    ds = wkw.Dataset.open(
        path.join(info.dataset_path, info.layer_name, str(info.mag)), info.header
    )
    return ds


def ensure_wkw(target_wkw_info):
    assert target_wkw_info.header is not None
    # Open will create the dataset if it doesn't exist yet
    target_wkw = open_wkw(target_wkw_info)
    target_wkw.close()


def cube_addresses(source_wkw_info):
    # Gathers all WKW cubes in the dataset
    with open_wkw(source_wkw_info) as source_wkw:
        wkw_addresses = list(parse_cube_file_name(f) for f in source_wkw.list_files())
        wkw_addresses.sort()
        return wkw_addresses


def parse_cube_file_name(filename):
    m = CUBE_REGEX.search(filename)
    return int(m.group(3)), int(m.group(2)), int(m.group(1))


def parse_scale(scale):
    try:
        scale = tuple(float(x) for x in scale.split(","))
        return scale
    except Exception:
        raise argparse.ArgumentTypeError("The scale could not be parsed")


def parse_bounding_box(bbox_str):
    try:
        return BoundingBox.from_csv(bbox_str)
    except Exception:
        raise argparse.ArgumentTypeError("The bounding box could not be parsed.")


def open_knossos(info):
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def add_verbose_flag(parser):
    parser.add_argument(
        "--silent", help="Silent output", dest="verbose", action="store_false"
    )

    parser.set_defaults(verbose=True)


def add_scale_flag(parser, required=True):
    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25). This is the size of one voxel in nm.",
        required=required,
        type=parse_scale,
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


def add_interpolation_flag(parser):
    parser.add_argument(
        "--interpolation_mode",
        "-i",
        help="Interpolation mode (median, mode, nearest, bilinear or bicubic)",
        default="default",
    )


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
        dataset_path: str,
        layer_name: str,
        dtype,
        origin: Union[Tuple[int, int, int], List[int]],
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        # file_len specifies, how many buckets written per dimension into a wkw cube. Using 32,
        # results in 1 GB/wkw file for 8-bit data
        file_len: int = DEFAULT_WKW_FILE_LEN,
        mag: Mag = Mag("1"),
    ):

        self.dataset_path = dataset_path
        self.layer_name = layer_name
        self.buffer_size = buffer_size

        layer_path = path.join(self.dataset_path, self.layer_name, mag.to_layer_name())

        self.dtype = dtype
        self.dataset = wkw.Dataset.open(
            layer_path, wkw.Header(dtype, file_len=file_len)
        )
        self.origin = origin

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

        assert (
            len(self.buffer) <= self.buffer_size
        ), "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."

        uniq_dtypes = set(map(lambda _slice: _slice.dtype, self.buffer))
        assert (
            len(uniq_dtypes) == 1
        ), "The buffer of BufferedSliceWriter contains slices with differing dtype."
        assert uniq_dtypes.pop() == self.dtype, (
            "The buffer of BufferedSliceWriter contains slices with a dtype "
            "which differs from the dtype with which the BufferedSliceWriter was instantiated."
        )

        logger.debug(
            "({}) Writing {} slices at position {}.".format(
                getpid(), len(self.buffer), self.buffer_start_z
            )
        )
        log_memory_consumption()

        try:
            origin_with_offset = list(self.origin)
            origin_with_offset[2] = self.buffer_start_z
            x_max = max(slice.shape[0] for slice in self.buffer)
            y_max = max(slice.shape[1] for slice in self.buffer)

            self.buffer = [
                np.pad(
                    slice,
                    mode="constant",
                    pad_width=[
                        (0, x_max - slice.shape[0]),
                        (0, y_max - slice.shape[1]),
                    ],
                )
                for slice in self.buffer
            ]

            data = np.concatenate(
                [np.expand_dims(slice, 2) for slice in self.buffer], axis=2
            )
            self.dataset.write(origin_with_offset, data)

        except Exception as exc:
            logger.error(
                "({}) An exception occurred in BufferedSliceWriter._write_buffer with {} "
                "slices at position {}. Original error is:\n{}:{}\n\nTraceback:".format(
                    getpid(),
                    len(self.buffer),
                    self.buffer_start_z,
                    type(exc).__name__,
                    exc,
                )
            )
            traceback.print_tb(exc.__traceback__)
            logger.error("\n")

            raise exc
        finally:
            self.buffer = []

    def close(self):

        self._write_buffer()
        self.dataset.close()

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _tb):
        self.close()


def log_memory_consumption(additional_output=""):
    pid = os.getpid()
    process = psutil.Process(pid)
    logging.info(
        "Currently consuming {:.2f} GB of memory ({:.2f} GB still available) "
        "in process {}. {}".format(
            process.memory_info().rss / 1024 ** 3,
            psutil.virtual_memory().available / 1024 ** 3,
            pid,
            additional_output,
        )
    )


def pad_or_crop_to_size_and_topleft(cube_data, target_size, target_topleft):
    """
    Given an numpy array and a target_size/target_topleft, the array
    will be padded so that it is within the bounding box descriped by topleft and size.
    If the input data is too large, the data will be cropped (evenly from opposite sides
    with the assumption that the most important data is in the center).
    """

    # Pad to size
    half_padding = (target_size - cube_data.shape) / 2
    half_padding = np.clip(half_padding, 0, None)
    left_padding = np.floor(half_padding).astype(np.uint32)
    right_padding = np.floor(half_padding).astype(np.uint32)

    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (left_padding[1], right_padding[1]),
            (left_padding[2], right_padding[2]),
            (0, 0),
        ),
    )

    # Potentially crop to size
    half_overflow = (cube_data.shape - target_size) / 2
    half_overflow = np.clip(half_overflow, 0, None)
    left_overflow = np.floor(half_overflow).astype(np.uint32)
    right_overflow = np.floor(half_overflow).astype(np.uint32)
    cube_data = cube_data[
        :,
        left_overflow[1] : cube_data.shape[1] - right_overflow[1],
        left_overflow[2] : cube_data.shape[2] - right_overflow[2],
        :,
    ]

    # Pad to topleft
    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (target_topleft[1], max(0, target_size[1] - cube_data.shape[1])),
            (target_topleft[2], max(0, target_size[2] - cube_data.shape[2])),
            (target_topleft[3], max(0, target_size[3] - cube_data.shape[3])),
        ),
    )

    return cube_data

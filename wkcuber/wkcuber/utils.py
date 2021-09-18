import re
from pathlib import Path
from types import TracebackType

import logging
import argparse

import wkw
import numpy as np
import os
import psutil
import traceback

from typing import (
    List,
    Tuple,
    Union,
    Iterable,
    Generator,
    Any,
    Optional,
    Type, cast, TYPE_CHECKING,
)
from glob import iglob
from collections import namedtuple
from multiprocessing import cpu_count
from os import path, getpid
from math import floor, ceil
from logging import getLogger
if TYPE_CHECKING:
    from webknossos.dataset import View
from wkcuber.api.bounding_box import BoundingBox

from .knossos import KnossosDataset
from .mag import Mag

from webknossos.dataset.defaults import DEFAULT_WKW_FILE_LEN
from webknossos.utils import *  # pylint: disable=unused-wildcard-import,wildcard-import

WkwDatasetInfo = namedtuple(
    "WkwDatasetInfo", ("dataset_path", "layer_name", "mag", "header")
)
KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
FallbackArgs = namedtuple("FallbackArgs", ("distribution_strategy", "jobs"))

BLOCK_LEN = 32
DEFAULT_WKW_VOXELS_PER_BLOCK = 32
CUBE_REGEX = re.compile(
    fr"z(\d+){re.escape(os.path.sep)}y(\d+){re.escape(os.path.sep)}x(\d+)(\.wkw)$"
)

logger = getLogger(__name__)


Vec3 = Union[Tuple[int, int, int], np.ndarray]


def open_wkw(info: WkwDatasetInfo) -> wkw.Dataset:
    ds = wkw.Dataset.open(
        str(info.dataset_path / info.layer_name / str(info.mag)), info.header
    )
    return ds


def ensure_wkw(target_wkw_info: WkwDatasetInfo) -> None:
    assert target_wkw_info.header is not None
    # Open will create the dataset if it doesn't exist yet
    target_wkw = open_wkw(target_wkw_info)
    target_wkw.close()


def cube_addresses(source_wkw_info: WkwDatasetInfo) -> List[Tuple[int, int, int]]:
    # Gathers all WKW cubes in the dataset
    with open_wkw(source_wkw_info) as source_wkw:
        wkw_addresses = list(parse_cube_file_name(f) for f in source_wkw.list_files())
        wkw_addresses.sort()
        return wkw_addresses


def parse_cube_file_name(filename: str) -> Tuple[int, int, int]:
    m = CUBE_REGEX.search(filename)
    if m is None:
        raise ValueError(f"Failed to parse cube file name {filename}")
    return int(m.group(3)), int(m.group(2)), int(m.group(1))


def parse_scale(scale: str) -> Tuple[float, ...]:
    try:
        return tuple(float(x) for x in scale.split(","))
    except Exception as e:
        raise argparse.ArgumentTypeError("The scale could not be parsed") from e


def parse_bounding_box(bbox_str: str) -> BoundingBox:
    try:
        return BoundingBox.from_csv(bbox_str)
    except Exception as e:
        raise argparse.ArgumentTypeError("The bounding box could not be parsed.") from e


def open_knossos(info: KnossosDatasetInfo) -> KnossosDataset:
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def add_verbose_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--silent", help="Silent output", dest="verbose", action="store_false"
    )

    parser.set_defaults(verbose=True)


def add_scale_flag(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--scale",
        "-s",
        help="Scale of the dataset (e.g. 11.2,11.2,25). This is the size of one voxel in nm.",
        required=required,
        type=parse_scale,
    )


def add_isotropic_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--isotropic",
        help="Activates isotropic downsampling. The default is anisotropic downsampling. "
        "Isotropic downsampling will always downsample each dimension with the factor 2.",
        dest="isotropic",
        default=None,
        action="store_true",
    )


def add_interpolation_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interpolation_mode",
        "-i",
        help="Interpolation mode (median, mode, nearest, bilinear or bicubic)",
        default="default",
    )


def add_sampling_mode_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sampling_mode",
        help="There are three different types: "
        "'anisotropic' - The next magnification is chosen so that the width, height and depth of a downsampled voxel assimilate. For example, if the z resolution is worse than the x/y resolution, z won't be downsampled in the first downsampling step(s). As a basis for this method, the scale from the datasource-properties.json is used. "
        "'isotropic' - Each dimension is downsampled equally. "
        "'constant_z' - The x and y dimensions are downsampled equally, but the z dimension remains the same.",
        default="anisotropic",
    )


def setup_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def find_files(
    source_path: str, extensions: Iterable[str]
) -> Generator[str, Any, None]:
    # Find all files with a matching file extension
    return (
        f
        for f in iglob(source_path, recursive=True)
        if any([f.lower().endswith(suffix) for suffix in extensions])
    )


def get_chunks(arr: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


# min_z and max_z are both inclusive
def get_regular_chunks(
    min_z: int, max_z: int, chunk_size: int
) -> Iterable[Iterable[int]]:
    i = floor(min_z / chunk_size) * chunk_size
    while i < ceil((max_z + 1) / chunk_size) * chunk_size:
        yield range(i, i + chunk_size)
        i += chunk_size


def add_distribution_flags(parser: argparse.ArgumentParser) -> None:
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


def add_batch_size_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of sections to buffer per job",
        type=int,
        default=BLOCK_LEN,
    )


class BufferedSliceWriter(object):
    """
    The BufferedSliceWriter buffers multiple slices before they are written to disk.
    The amount of slices that get buffered is specified by `buffer_size`.
    As soon as the buffer is full, the data gets written to disk.

    The user can specify along which dimension the data is sliced by using the parameter `dimension`.
    To slice along the x-axis use `0`, for the y-axis use `1`, or for the z-axis use `2` (default: dimension=2).

    This class is supposed to be used with a context manager (see example below).
    If the buffer is non empty after the user finished writing (i.e. the number of written slices is not a multiple of `buffer_size`),
    exiting the context will automatically write the buffer to disk.
    Entering the context returns a generator with consumes slices (np.ndarray).
    Note: this generator pattern requires to start the generator by sending `None` to it.

    Usage:
    data_cube = ...
    with BufferedSliceWriter(...) as writer:
        writer.send(None)  # to start the generator
        for data_slice in data_cube:
            writer.send(data_slice)

    """
    def __init__(
        self,
        view: View,
        offset: Vec3,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2  # z
    ) -> None:
        """
        view : datasource
        offset : specifies the offset of the data to write (relative to the `view`)
        buffer_size : the number of slices that are read at once
        dimension : specifies along which axis the data is sliced (0=x; 1=y; 2=z)

        The size is in the magnification of the `view`.
        """
        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        self.offset = offset
        self.dimension = dimension

        assert 0 <= dimension <= 2

        self.buffer: List[np.ndarray] = []
        self.current_slice: Optional[int] = None
        self.buffer_start_slice: Optional[int] = None

    def _write_buffer(self) -> None:
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
                getpid(), len(self.buffer), self.buffer_start_slice
            )
        )
        log_memory_consumption()

        try:
            assert (
                self.buffer_start_slice is not None
            ), "Failed to write buffer: The buffer_start_slice is not set."
            buffer_start = [0, 0, 0]
            buffer_start[self.dimension] = self.buffer_start_slice
            offset = cast(Tuple[int, int, int], tuple([off + buff_off for off, buff_off in zip(self.offset, buffer_start)]))
            max_width = max(slice.shape[-2] for slice in self.buffer)
            max_height = max(slice.shape[-1] for slice in self.buffer)

            self.buffer = [
                np.pad(
                    slice,
                    mode="constant",
                    pad_width=[
                        (0, 0),
                        (0, max_width - slice.shape[-2]),
                        (0, max_height - slice.shape[-1]),
                    ],
                )
                for slice in self.buffer
            ]

            data = np.concatenate(
                [np.expand_dims(slice, self.dimension+1) for slice in self.buffer], axis=self.dimension+1
            )
            self.view.write(data, offset)

        except Exception as exc:
            logger.error(
                "({}) An exception occurred in BufferedSliceWriter._write_buffer with {} "
                "slices at position {}. Original error is:\n{}:{}\n\nTraceback:".format(
                    getpid(),
                    len(self.buffer),
                    self.buffer_start_slice,
                    type(exc).__name__,
                    exc,
                )
            )
            traceback.print_tb(exc.__traceback__)
            logger.error("\n")

            raise exc
        finally:
            self.buffer = []

    def get_slice_generator(self) -> Generator[None, np.ndarray, None]:
        """Reads a WKW data, returns slices in [y, x] shape."""
        current_slice = 0
        while True:
            data = yield  # Data gets send from the user
            if len(self.buffer) == 0:
                self.buffer_start_slice = current_slice
            if len(data.shape) == 2:
                # The input data might contain channel data or not.
                # Bringing it into the same shape simplifies the code
                data = np.expand_dims(data, axis=0)
            self.buffer.append(data)
            current_slice += 1

            if current_slice % self.buffer_size == 0:
                self._write_buffer()

    def __enter__(self) -> Generator[None, np.ndarray, None]:
        return self.get_slice_generator()

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self._write_buffer()


class BufferedSliceReader(object):
    """
    The BufferedSliceReader reads multiple slices from disk at once and buffers the data.
    The amount of slices that get buffered is specified by `buffer_size`.
    The slices are then yielded to the user separately.

    The user can specify along which dimension the data is sliced by using the parameter `dimension`.
    To slice along the x-axis use `0`, for the y-axis use `1`, or for the z-axis use `2` (default: dimension=2).

    This class is supposed to be used with a context manager (see example below).
    Entering the context returns a generator with yields slices (np.ndarray).

    Usage:
    with BufferedSliceReader(...) as reader:
        for slice_data in reader:
            ...

    """
    def __init__(
        self,
        view: View,
        offset: Vec3,
        size: Vec3,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2  # z
    ) -> None:
        """
        view : datasource
        offset : specifies the offset of the data to read (relative to the `view`)
        size : specifies the size of the data to read
        buffer_size : the number of slices that are read at once
        dimension : specifies along which axis the data is sliced (0=x; 1=y; 2=z)

        The size and offset are in the magnification of the `view`.
        """

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        assert 0 <= dimension <= 2
        self.dimension = dimension
        bounding_box = BoundingBox(view.global_offset, view.size)
        self.target_bbox = bounding_box.intersected_with(BoundingBox(view.global_offset, size).offset(cast(Tuple[int, int, int], tuple(offset))))

    def _get_slice_generator(self) -> Generator[np.ndarray, None, None]:
        for batch in get_chunks(list(range(
            self.target_bbox.topleft[self.dimension],
            self.target_bbox.bottomright[self.dimension]
        )), self.buffer_size):
            n_slices = len(batch)
            batch_start_idx = batch[0]


            assert (
                n_slices <= self.buffer_size
            ), f"n_slices should at most be batch_size, but {n_slices} > {self.buffer_size}"

            bbox_offset = self.target_bbox.topleft.tolist()
            bbox_size = self.target_bbox.size.tolist()

            buffer_bounding_box = BoundingBox.from_tuple2(
                (
                    bbox_offset[:self.dimension] + [batch_start_idx] + bbox_offset[self.dimension+1:],
                    bbox_size[:self.dimension] + [n_slices] + bbox_size[self.dimension+1:],
                )
            )

            logger.debug(f"({getpid()}) Reading {n_slices} slices at position {batch_start_idx}.")
            negative_view_offset = cast(Tuple[int, int, int], tuple([-o for o in self.view.global_offset]))  # this needs to be subtracted from the buffer_bounding_box because the view expects a relative offset
            data = self.view.read_bbox(buffer_bounding_box.offset(negative_view_offset))

            for current_slice in np.rollaxis(data, self.dimension+1):  # The '+1' is important because the first dimension is the channel
                yield current_slice

    def __enter__(self) -> Generator[np.ndarray, None, None]:
        return self._get_slice_generator()

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        ...


def log_memory_consumption(additional_output: str = "") -> None:
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


def pad_or_crop_to_size_and_topleft(
    cube_data: np.ndarray, target_size: np.ndarray, target_topleft: np.ndarray
) -> np.ndarray:
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


def convert_mag1_offset(
    mag1_offset: Union[List, np.ndarray], target_mag: Mag
) -> np.ndarray:
    return np.array(mag1_offset) // target_mag.as_np()  # floor div


def get_executor_args(global_args: argparse.Namespace) -> argparse.Namespace:
    executor_args = argparse.Namespace()
    executor_args.jobs = global_args.jobs
    executor_args.distribution_strategy = global_args.distribution_strategy
    executor_args.job_resources = global_args.job_resources
    return executor_args

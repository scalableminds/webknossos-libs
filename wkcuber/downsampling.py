import time
import logging
import wkw
import re
import numpy as np
from argparse import ArgumentParser
from math import floor
from os import path, listdir
from scipy.stats import mode
from scipy.ndimage.interpolation import zoom
from itertools import product
from functools import lru_cache
from enum import Enum

from .utils import (
    add_jobs_flag,
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ParallelExecutor,
    pool_get_lock,
)

DEFAULT_EDGE_LEN = 256


class InterpolationModes(Enum):
    MEDIAN = 0
    MODE = 1
    NEAREST = 2
    BILINEAR = 3
    BICUBIC = 4
    MAX = 5
    MIN = 6


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("path", help="Directory containing the dataset.")

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--interpolation_mode",
        "-i",
        help="Interpolation mode (median, mode, nearest, bilinear or bicubic)",
        default="default",
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8",
    )

    parser.add_argument(
        "--max", "-m", help="Max resolution to be downsampled", default=512
    )

    parser.add_argument(
        "--buffer_cube_size",
        "-b",
        help="Size of buffered cube to be downsampled (i.e. buffer cube edge length)",
        default=DEFAULT_EDGE_LEN,
    )

    add_jobs_flag(parser)
    add_verbose_flag(parser)

    return parser


def cube_addresses(source_wkw_info, cube_edge_len):
    # Traverse all WKW cubes in the dataset in order to
    # find all available cubes of size `cube_edge_len`^3
    with open_wkw(source_wkw_info) as source_wkw:
        wkw_cubelength = source_wkw.header.file_len * source_wkw.header.block_len
        factor = wkw_cubelength // cube_edge_len

        def parse_cube_file_name(filename):
            CUBE_REGEX = re.compile("z(\d+)/y(\d+)/x(\d+)(\.wkw)$")
            m = CUBE_REGEX.search(filename)
            return (int(m.group(3)), int(m.group(2)), int(m.group(1)))

        wkw_addresses = list(parse_cube_file_name(f) for f in source_wkw.list_files())

        cube_addresses = []
        for wkw_x, wkw_y, wkw_z in wkw_addresses:
            x_dims = list(range(wkw_x * factor, (wkw_x + 1) * factor))
            y_dims = list(range(wkw_y * factor, (wkw_y + 1) * factor))
            z_dims = list(range(wkw_z * factor, (wkw_z + 1) * factor))
            cube_addresses += product(x_dims, y_dims, z_dims)

        cube_addresses.sort()
        return cube_addresses


def downsample(
    source_wkw_info,
    target_wkw_info,
    source_mag,
    target_mag,
    interpolation_mode,
    cube_edge_len,
    jobs,
):
    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(target_mag, source_mag))

    factor = int(target_mag / source_mag)
    # Detect the cubes that we want to downsample
    source_cube_addresses = cube_addresses(source_wkw_info, cube_edge_len)
    target_cube_addresses = list(
        set(tuple(x // factor for x in xyz) for xyz in source_cube_addresses)
    )
    target_cube_addresses.sort()
    logging.debug(
        "Found source cubes: count={} size={} min={} max={}".format(
            len(source_cube_addresses),
            (cube_edge_len,) * 3,
            min(source_cube_addresses),
            max(source_cube_addresses),
        )
    )
    logging.debug(
        "Found target cubes: count={} size={} min={} max={}".format(
            len(target_cube_addresses),
            (cube_edge_len,) * 3,
            min(target_cube_addresses),
            max(target_cube_addresses),
        )
    )

    with ParallelExecutor(jobs) as pool:
        for target_cube_xyz in target_cube_addresses:
            pool.submit(
                downsample_cube_job,
                source_wkw_info,
                target_wkw_info,
                factor,
                interpolation_mode,
                cube_edge_len,
                target_cube_xyz,
            )

    logging.info("Mag {0} succesfully cubed".format(target_mag))


def downsample_cube_job(
    source_wkw_info,
    target_wkw_info,
    factor,
    interpolation_mode,
    cube_edge_len,
    target_cube_xyz,
):
    try:
        logging.debug("Downsampling {}".format(target_cube_xyz))

        with open_wkw(source_wkw_info) as source_wkw, open_wkw(
            target_wkw_info, pool_get_lock()
        ) as target_wkw:
            target_offset = tuple(a * cube_edge_len for a in target_cube_xyz)
            source_offset = tuple(a * factor for a in target_offset)

            ref_time = time.time()
            # Read source buffer
            cube_buffer = source_wkw.read(source_offset, (cube_edge_len * factor,) * 3)
            assert cube_buffer.shape[0] == 1, "Only single-channel data is supported"
            cube_buffer = cube_buffer[0]

            if np.all(cube_buffer == 0):
                logging.debug("Skipping empty cube {}".format(target_cube_xyz))
            # Downsample the buffer
            cube_data = downsample_cube(cube_buffer, factor, interpolation_mode)

            # Write the downsampled buffer to target
            target_wkw.write(target_offset, cube_data)

        logging.debug(
            "Downsampling of {} took {:.8f}s".format(
                target_cube_xyz, time.time() - ref_time
            )
        )
    except Exception as exc:
        logging.error("Downsampling of {} failed with {}".format(target_cube_xyz, exc))
        raise exc


def non_linear_filter_3d(data, factor, func):
    ds = data.shape
    assert not any((d % factor > 0 for d in ds))
    data = data.reshape((ds[0], factor, ds[1] // factor, ds[2]), order="F")
    data = data.swapaxes(0, 1)
    data = data.reshape(
        (factor * factor, ds[0] * ds[1] // (factor * factor), factor, ds[2] // factor),
        order="F",
    )
    data = data.swapaxes(2, 1)
    data = data.reshape(
        (
            factor * factor * factor,
            (ds[0] * ds[1] * ds[2]) // (factor * factor * factor),
        ),
        order="F",
    )
    data = func(data)
    data = data.reshape((ds[0] // factor, ds[1] // factor, ds[2] // factor), order="F")
    return data


def linear_filter_3d(data, factor, order):
    ds = data.shape
    assert not any((d % factor > 0 for d in ds))
    return zoom(
        data,
        1 / factor,
        output=data.dtype,
        # 0: nearest
        # 1: bilinear
        # 2: bicubic
        order=order,
        # this does not mean nearest interpolation,
        # it corresponds to how the borders are treated.
        mode="nearest",
        prefilter=True,
    )


def _max(x):
    return np.max(x, axis=0)


def _min(x):
    return np.min(x, axis=0)


def _median(x):
    return np.median(x, axis=0).astype(x.dtype)


def _mode(x):
    return mode(x, axis=0, nan_policy="omit")[0][0]


def downsample_cube(cube_buffer, factor, interpolation_mode):
    if interpolation_mode == InterpolationModes.MODE:
        return non_linear_filter_3d(cube_buffer, factor, _mode)
    elif interpolation_mode == InterpolationModes.MEDIAN:
        return non_linear_filter_3d(cube_buffer, factor, _median)
    elif interpolation_mode == InterpolationModes.NEAREST:
        return linear_filter_3d(cube_buffer, factor, 0)
    elif interpolation_mode == InterpolationModes.BILINEAR:
        return linear_filter_3d(cube_buffer, factor, 1)
    elif interpolation_mode == InterpolationModes.BICUBIC:
        return linear_filter_3d(cube_buffer, factor, 2)
    elif interpolation_mode == InterpolationModes.MAX:
        return non_linear_filter_3d(cube_buffer, factor, _max)
    elif interpolation_mode == InterpolationModes.MIN:
        return non_linear_filter_3d(cube_buffer, factor, _min)
    else:
        raise Exception("Invalid interpolation mode: {}".format(interpolation_mode))


def downsample_mag(
    path,
    layer_name,
    source_mag,
    target_mag,
    dtype="uint8",
    interpolation_mode="default",
    cube_edge_len=DEFAULT_EDGE_LEN,
    jobs=1,
):
    if interpolation_mode == "default":
        interpolation_mode = (
            InterpolationModes.MEDIAN
            if layer_name == "color"
            else InterpolationModes.MODE
        )
    else:
        interpolation_mode = InterpolationModes[interpolation_mode.upper()]

    source_wkw_info = WkwDatasetInfo(path, layer_name, dtype, source_mag)
    target_wkw_info = WkwDatasetInfo(path, layer_name, dtype, target_mag)
    downsample(
        source_wkw_info,
        target_wkw_info,
        source_mag,
        target_mag,
        interpolation_mode,
        cube_edge_len,
        jobs,
    )


def downsample_mags(
    path, layer_name, max_mag, dtype, interpolation_mode, cube_edge_len, jobs
):
    target_mag = 2
    while target_mag <= int(max_mag):
        source_mag = target_mag // 2
        downsample_mag(
            path,
            layer_name,
            source_mag,
            target_mag,
            dtype,
            interpolation_mode,
            cube_edge_len,
            jobs,
        )
        target_mag = target_mag * 2


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    downsample_mags(
        args.path,
        args.layer_name,
        int(args.max),
        args.dtype,
        args.interpolation_mode,
        int(args.buffer_cube_size),
        int(args.jobs),
    )

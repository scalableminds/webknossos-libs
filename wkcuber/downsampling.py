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
    time_start,
    time_stop,
)

DEFAULT_EDGE_LEN = 256
CUBE_REGEX = re.compile("z(\d+)/y(\d+)/x(\d+)(\.wkw)$")


def parse_cube_file_name(filename):
    m = CUBE_REGEX.search(filename)
    return (int(m.group(3)), int(m.group(2)), int(m.group(1)))


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
        "--max", "-m", help="Max resolution to be downsampled", type=int, default=512
    )

    parser.add_argument(
        "--buffer_cube_size",
        "-b",
        help="Size of buffered cube to be downsampled (i.e. buffer cube edge length)",
        type=int,
        default=DEFAULT_EDGE_LEN,
    )

    parser.add_argument(
        "--compress", action="store_true", help="Compress data during downsampling"
    )

    add_jobs_flag(parser)
    add_verbose_flag(parser)

    return parser


def cube_addresses(source_wkw_info, cube_edge_len):
    # Traverse all WKW cubes in the dataset in order to
    # find all available cubes of size `cube_edge_len`^3
    with open_wkw(source_wkw_info) as source_wkw:
        wkw_cubelength = source_wkw.header.file_len * source_wkw.header.block_len
        buffer_len_factor = wkw_cubelength // cube_edge_len

        wkw_addresses = list(parse_cube_file_name(f) for f in source_wkw.list_files())

        cube_addresses = []
        for wkw_x, wkw_y, wkw_z in wkw_addresses:
            x_dims = list(
                range(wkw_x * buffer_len_factor, (wkw_x + 1) * buffer_len_factor)
            )
            y_dims = list(
                range(wkw_y * buffer_len_factor, (wkw_y + 1) * buffer_len_factor)
            )
            z_dims = list(
                range(wkw_z * buffer_len_factor, (wkw_z + 1) * buffer_len_factor)
            )
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
    compress,
):
    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(target_mag, source_mag))

    mag_factor = int(target_mag / source_mag)
    # Detect the cubes that we want to downsample
    source_cube_addresses = cube_addresses(source_wkw_info, cube_edge_len)
    target_cube_addresses = list(
        set(tuple(x // mag_factor for x in xyz) for xyz in source_cube_addresses)
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
                mag_factor,
                interpolation_mode,
                cube_edge_len,
                target_cube_xyz,
                compress,
            )

    logging.info("Mag {0} succesfully cubed".format(target_mag))


def downsample_cube_job(
    source_wkw_info,
    target_wkw_info,
    mag_factor,
    interpolation_mode,
    cube_edge_len,
    target_cube_xyz,
    compress,
):
    try:
        header_block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        with open_wkw(source_wkw_info) as source_wkw, open_wkw(
            target_wkw_info, pool_get_lock(), header_block_type
        ) as target_wkw:
            wkw_cubelength = source_wkw.header.file_len * source_wkw.header.block_len

            file_buffer = np.zeros((wkw_cubelength,) * 3, target_wkw_info.dtype)
            tile_length = cube_edge_len
            tile_count_per_dim = wkw_cubelength // tile_length
            assert (
                wkw_cubelength % cube_edge_len == 0
            ), "buffer_cube_size must be a divisor of wkw cube length"

            tile_indices = list(range(0, tile_count_per_dim))
            tiles = product(tile_indices, tile_indices, tile_indices)
            file_offset = wkw_cubelength * np.array(target_cube_xyz)

            for tile in tiles:
                time_start("process tile")

                target_offset = np.array(
                    tile
                ) * tile_length + wkw_cubelength * np.array(target_cube_xyz)
                source_offset = mag_factor * target_offset
                logging.debug("        tile {}".format(tile))
                logging.debug("        target_offset {}".format(target_offset))
                logging.debug("        source_offset {}".format(source_offset))

                # Read source buffer
                time_start("wkw::read")
                cube_buffer = source_wkw.read(
                    source_offset,
                    (wkw_cubelength * mag_factor // tile_count_per_dim,) * 3,
                )
                time_stop("wkw::read")
                assert (
                    cube_buffer.shape[0] == 1
                ), "Only single-channel data is supported"
                cube_buffer = cube_buffer[0]

                if np.all(cube_buffer == 0):
                    logging.debug(
                        "        Skipping empty cube {} (tile {})".format(
                            target_cube_xyz, tile
                        )
                    )
                else:
                    # Downsample the buffer

                    time_start("apply downsample")
                    data_cube = downsample_cube(
                        cube_buffer, mag_factor, interpolation_mode
                    )
                    logging.debug("before downsample_cube {}".format(data_cube.shape))
                    logging.debug("data_cube.shape: {}".format(data_cube.shape))

                    buffer_offset = target_offset - file_offset
                    buffer_end = buffer_offset + tile_length

                    file_buffer[
                        buffer_offset[0] : buffer_end[0],
                        buffer_offset[1] : buffer_end[1],
                        buffer_offset[2] : buffer_end[2],
                    ] = data_cube
                    time_stop("apply downsample")

                    time_stop("process tile")
                    logging.debug("  ")

            time_start("Downsampling of {}".format(target_cube_xyz))
            # Write the downsampled buffer to target
            target_wkw.write(file_offset, file_buffer)
            time_stop("Downsampling of {}".format(target_cube_xyz))

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
    compress=False,
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
        compress,
    )


def downsample_mags(
    path, layer_name, max_mag, dtype, interpolation_mode, cube_edge_len, jobs, compress
):
    target_mag = 2
    while target_mag <= max_mag:
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
            compress,
        )
        target_mag = target_mag * 2


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    downsample_mags(
        args.path,
        args.layer_name,
        args.max,
        args.dtype,
        args.interpolation_mode,
        args.buffer_cube_size,
        args.jobs,
        args.compress,
    )

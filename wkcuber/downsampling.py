import logging
import wkw
import re
import numpy as np
from argparse import ArgumentParser
from math import floor, log2
from os import path, listdir
from scipy.stats import mode
from scipy.ndimage.interpolation import zoom
from itertools import product
from functools import lru_cache
from enum import Enum
from .mag import Mag

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
CUBE_REGEX = re.compile(r"z(\d+)/y(\d+)/x(\d+)(\.wkw)$")


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
        "--from_mag",
        "--from",
        "-f",
        help="Resolution to base downsampling on",
        type=int,
        default=1,
    )

    # Either provide the maximum resolution to be downsampled OR a specific, anisotropic magnification.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max", "-m", help="Max resolution to be downsampled", type=int, default=512
    )
    group.add_argument(
        "--anisotropic_target_mag",
        help="Specify an anisotropic target magnification which should be created (e.g., --anisotropic_target_mag 2-2-1)",
        type=str,
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


def cube_addresses(source_wkw_info):
    # Gathers all WKW cubes in the dataset
    with open_wkw(source_wkw_info) as source_wkw:
        wkw_addresses = list(parse_cube_file_name(f) for f in source_wkw.list_files())
        wkw_addresses.sort()
        return wkw_addresses


def downsample(
    source_wkw_info,
    target_wkw_info,
    source_mag: Mag,
    target_mag: Mag,
    interpolation_mode,
    cube_edge_len,
    jobs,
    compress,
):

    assert source_mag < target_mag
    logging.info("Downsampling mag {} from mag {}".format(target_mag, source_mag))

    mag_factors = [
        t // s for (t, s) in zip(target_mag.to_array(), source_mag.to_array())
    ]
    # Detect the cubes that we want to downsample
    source_cube_addresses = cube_addresses(source_wkw_info)

    target_cube_addresses = list(
        set(
            tuple(dim // mag_factor for (dim, mag_factor) in zip(xyz, mag_factors))
            for xyz in source_cube_addresses
        )
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
                mag_factors,
                interpolation_mode,
                cube_edge_len,
                target_cube_xyz,
                compress,
            )

    logging.info("Mag {0} succesfully cubed".format(target_mag))


def downsample_cube_job(
    source_wkw_info,
    target_wkw_info,
    mag_factors,
    interpolation_mode,
    cube_edge_len,
    target_cube_xyz,
    compress,
):
    logging.info("Downsampling of {}".format(target_cube_xyz))

    try:
        time_start("Downsampling of {}".format(target_cube_xyz))
        header_block_type = (
            wkw.Header.BLOCK_TYPE_LZ4HC if compress else wkw.Header.BLOCK_TYPE_RAW
        )

        with open_wkw(source_wkw_info) as source_wkw:
            num_channels = source_wkw.header.num_channels
            with open_wkw(
                target_wkw_info,
                pool_get_lock(),
                block_type=header_block_type,
                num_channels=num_channels,
            ) as target_wkw:
                wkw_cubelength = (
                    source_wkw.header.file_len * source_wkw.header.block_len
                )
                shape = (num_channels,) + (wkw_cubelength,) * 3
                file_buffer = np.zeros(shape, target_wkw_info.dtype)
                tile_length = cube_edge_len
                tile_count_per_dim = wkw_cubelength // tile_length
                assert (
                    wkw_cubelength % cube_edge_len == 0
                ), "buffer_cube_size must be a divisor of wkw cube length"

                tile_indices = list(range(0, tile_count_per_dim))
                tiles = product(tile_indices, tile_indices, tile_indices)
                file_offset = wkw_cubelength * np.array(target_cube_xyz)

                for tile in tiles:
                    target_offset = np.array(
                        tile
                    ) * tile_length + wkw_cubelength * np.array(target_cube_xyz)
                    source_offset = mag_factors * target_offset

                    # Read source buffer
                    cube_buffer_channels = source_wkw.read(
                        source_offset,
                        (wkw_cubelength * np.array(mag_factors) // tile_count_per_dim),
                    )

                    for channel_index in range(num_channels):
                        cube_buffer = cube_buffer_channels[channel_index]

                        if np.all(cube_buffer == 0):
                            logging.debug(
                                "        Skipping empty cube {} (tile {})".format(
                                    target_cube_xyz, tile
                                )
                            )
                        else:
                            # Downsample the buffer

                            data_cube = downsample_cube(
                                cube_buffer, mag_factors, interpolation_mode
                            )

                            buffer_offset = target_offset - file_offset
                            buffer_end = buffer_offset + tile_length

                            file_buffer[
                                channel_index,
                                buffer_offset[0] : buffer_end[0],
                                buffer_offset[1] : buffer_end[1],
                                buffer_offset[2] : buffer_end[2],
                            ] = data_cube

                # Write the downsampled buffer to target
                target_wkw.write(file_offset, file_buffer)
        time_stop("Downsampling of {}".format(target_cube_xyz))

    except Exception as exc:
        logging.error("Downsampling of {} failed with {}".format(target_cube_xyz, exc))
        raise exc


def non_linear_filter_3d(data, factors, func):
    ds = data.shape
    assert not any((d % factor > 0 for (d, factor) in zip(ds, factors)))
    data = data.reshape((ds[0], factors[1], ds[1] // factors[1], ds[2]), order="F")
    data = data.swapaxes(0, 1)
    data = data.reshape(
        (
            factors[0] * factors[1],
            ds[0] * ds[1] // (factors[0] * factors[1]),
            factors[2],
            ds[2] // factors[2],
        ),
        order="F",
    )
    data = data.swapaxes(2, 1)
    data = data.reshape(
        (
            factors[0] * factors[1] * factors[2],
            (ds[0] * ds[1] * ds[2]) // (factors[0] * factors[1] * factors[2]),
        ),
        order="F",
    )
    data = func(data)
    data = data.reshape(
        (ds[0] // factors[0], ds[1] // factors[1], ds[2] // factors[2]), order="F"
    )
    return data


def linear_filter_3d(data, factors, order):
    factors = np.array(factors)

    if not np.all(factors == factors[0]):
        logging.debug(
            "the selected filtering strategy does not support anisotropic downsampling. Selecting {} as uniform downsampling factor".format(
                factors[0]
            )
        )
    factor = factors[0]

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


def downsample_cube(cube_buffer, factors, interpolation_mode):
    if interpolation_mode == InterpolationModes.MODE:
        return non_linear_filter_3d(cube_buffer, factors, _mode)
    elif interpolation_mode == InterpolationModes.MEDIAN:
        return non_linear_filter_3d(cube_buffer, factors, _median)
    elif interpolation_mode == InterpolationModes.NEAREST:
        return linear_filter_3d(cube_buffer, factors, 0)
    elif interpolation_mode == InterpolationModes.BILINEAR:
        return linear_filter_3d(cube_buffer, factors, 1)
    elif interpolation_mode == InterpolationModes.BICUBIC:
        return linear_filter_3d(cube_buffer, factors, 2)
    elif interpolation_mode == InterpolationModes.MAX:
        return non_linear_filter_3d(cube_buffer, factors, _max)
    elif interpolation_mode == InterpolationModes.MIN:
        return non_linear_filter_3d(cube_buffer, factors, _min)
    else:
        raise Exception("Invalid interpolation mode: {}".format(interpolation_mode))


def downsample_mag(
    path,
    layer_name,
    source_mag: Mag,
    target_mag: Mag,
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

    source_wkw_info = WkwDatasetInfo(
        path, layer_name, dtype, source_mag.to_layer_name()
    )
    target_wkw_info = WkwDatasetInfo(
        path, layer_name, dtype, target_mag.to_layer_name()
    )
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
    path,
    layer_name,
    from_mag: Mag,
    max_mag: Mag,
    dtype,
    interpolation_mode,
    cube_edge_len,
    jobs,
    compress,
):
    target_mag = from_mag.scaled_by(2)
    while target_mag <= max_mag:
        source_mag = target_mag.divided_by(2)
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
        target_mag.scale_by(2)


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    from_mag = Mag(args.from_mag)
    max_mag = Mag(args.max)
    if args.anisotropic_target_mag:
        anisotropic_target_mag = Mag(args.anisotropic_target_mag)

        downsample_mag(
            args.path,
            args.layer_name,
            from_mag,
            anisotropic_target_mag,
            args.dtype,
            args.interpolation_mode,
            args.buffer_cube_size,
            args.jobs,
            args.compress,
        )
    else:
        downsample_mags(
            args.path,
            args.layer_name,
            from_mag,
            max_mag,
            args.dtype,
            args.interpolation_mode,
            args.buffer_cube_size,
            args.jobs,
            args.compress,
        )

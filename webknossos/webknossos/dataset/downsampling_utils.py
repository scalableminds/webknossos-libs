import logging
import math
from enum import Enum
from itertools import product
from typing import Callable, List, Optional, Tuple, cast

import numpy as np
from scipy.ndimage import zoom
from wkw import wkw

from webknossos.geometry import Mag, Vec3Int, Vec3IntLike
from webknossos.utils import time_start, time_stop

from .view import View


class SamplingModes:
    ANISOTROPIC = "anisotropic"
    ISOTROPIC = "isotropic"
    CONSTANT_Z = "constant_z"


class InterpolationModes(Enum):
    MEDIAN = 0
    MODE = 1
    NEAREST = 2
    BILINEAR = 3
    BICUBIC = 4
    MAX = 5
    MIN = 6


DEFAULT_EDGE_LEN = 256


def determine_buffer_edge_len(dataset: wkw.Dataset) -> int:
    return min(DEFAULT_EDGE_LEN, dataset.header.file_len * dataset.header.block_len)


def calculate_mags_to_downsample(
    from_mag: Mag, max_mag: Mag, scale: Optional[Tuple[float, float, float]]
) -> List[Mag]:
    assert np.all(from_mag.to_np() <= max_mag.to_np())
    mags = []
    current_mag = from_mag
    while current_mag < max_mag:
        if scale is None:
            # In case the sampling mode is CONSTANT_Z or ISOTROPIC:
            current_mag = Mag(np.minimum(current_mag.to_np() * 2, max_mag.to_np()))
        else:
            # In case the sampling mode is ANISOTROPIC:
            current_size = current_mag.to_np() * np.array(scale)
            min_value = np.min(current_size)
            min_value_bitmask = np.array(current_size == min_value)
            factor = min_value_bitmask + 1

            # Calculate the two potential magnifications.
            # Either, double all components or only double the smallest component.
            all_scaled = current_size * 2
            min_scaled = current_size * factor  # only multiply the smallest dimension

            all_scaled_ratio = np.max(all_scaled) / np.min(all_scaled)
            min_scaled_ratio = np.max(min_scaled) / np.min(min_scaled)

            # The smaller the ratio between the smallest dimension and the largest dimension, the better.
            if all_scaled_ratio < min_scaled_ratio:
                # Multiply all dimensions with "2"
                current_mag = Mag(np.minimum(current_mag.to_np() * 2, max_mag.to_np()))
            else:
                # Multiply only the minimal dimension by "2".
                current_mag = Mag(
                    np.minimum(current_mag.to_np() * factor, max_mag.to_np())
                )

        mags += [current_mag]

    return mags


def calculate_mags_to_upsample(
    from_mag: Mag, min_mag: Mag, scale: Optional[Tuple[float, float, float]]
) -> List[Mag]:
    return list(reversed(calculate_mags_to_downsample(min_mag, from_mag, scale)))[
        1:
    ] + [min_mag]


def calculate_default_max_mag(dataset_size: Vec3IntLike) -> Mag:
    dataset_size = Vec3Int(dataset_size)
    # The lowest mag should have a size of ~ 100vx**2 per slice
    max_x_y = max(dataset_size[0], dataset_size[1])
    # highest power of 2 larger (or equal) than max_x_y divided by 100
    # The calculated factor will be used for x, y and z here. If anisotropic downsampling takes place,
    # the dimensions can still be downsampled independently according to the scale.
    return Mag(max(2 ** math.ceil(math.log(max_x_y / 100, 2)), 4))  # at least 4


def parse_interpolation_mode(
    interpolation_mode: str, layer_name: str
) -> InterpolationModes:
    if interpolation_mode.upper() == "DEFAULT":
        return (
            InterpolationModes.MODE
            if layer_name == "segmentation"
            else InterpolationModes.MEDIAN
        )
    else:
        return InterpolationModes[interpolation_mode.upper()]


def linear_filter_3d(data: np.ndarray, factors: List[int], order: int) -> np.ndarray:
    factors_np = np.array(factors)

    if not np.all(factors_np == factors[0]):
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


def non_linear_filter_3d(
    data: np.ndarray, factors: List[int], func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
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


def _max(x: np.ndarray) -> np.ndarray:
    return np.max(x, axis=0)


def _min(x: np.ndarray) -> np.ndarray:
    return np.min(x, axis=0)


def _median(x: np.ndarray) -> np.ndarray:
    return np.median(x, axis=0).astype(x.dtype)


def _mode(x: np.ndarray) -> np.ndarray:
    """
    Fast mode implementation from: https://stackoverflow.com/a/35674754
    """
    # Check inputs
    ndim = x.ndim
    axis = 0
    # Sort array
    sort = np.sort(x, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = (
        np.concatenate(
            [
                np.zeros(shape=shape, dtype="bool"),
                np.diff(sort, axis=axis) == 0,
                np.zeros(shape=shape, dtype="bool"),
            ],
            axis=axis,
        )
        .transpose(transpose)
        .ravel()
    )
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape_array = np.array(sort.shape)
    shape_array[axis] += 1
    shape_array = shape_array[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape_array).transpose(transpose)[tuple(slices)] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[tuple(index)]


def downsample_unpadded_data(
    buffer: np.ndarray, target_mag: Mag, interpolation_mode: InterpolationModes
) -> np.ndarray:
    logging.info(
        f"Downsampling buffer of size {buffer.shape} to mag {target_mag.to_layer_name()}"
    )
    target_mag_np = np.array(target_mag.to_list())
    current_dimension_size = np.array(buffer.shape[1:])
    padding_size_for_downsampling = (
        target_mag_np - (current_dimension_size % target_mag_np) % target_mag_np
    )
    padding_size_for_downsampling = list(zip([0, 0, 0], padding_size_for_downsampling))
    buffer = np.pad(
        buffer, pad_width=[(0, 0)] + padding_size_for_downsampling, mode="constant"
    )
    dimension_decrease = np.array([1] + target_mag.to_list())
    downsampled_buffer_shape = np.array(buffer.shape) // dimension_decrease
    downsampled_buffer = np.empty(dtype=buffer.dtype, shape=downsampled_buffer_shape)
    for channel in range(buffer.shape[0]):
        downsampled_buffer[channel] = downsample_cube(
            buffer[channel], target_mag.to_list(), interpolation_mode
        )
    return downsampled_buffer


def downsample_cube(
    cube_buffer: np.ndarray, factors: List[int], interpolation_mode: InterpolationModes
) -> np.ndarray:
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


def downsample_cube_job(
    args: Tuple[View, View, int],
    mag_factors: List[int],
    interpolation_mode: InterpolationModes,
    buffer_edge_len: int,
    job_count_per_log: int,
) -> None:
    (source_view, target_view, i) = args
    use_logging = i % job_count_per_log == 0

    if use_logging:
        logging.info(f"Downsampling of {target_view.global_offset}")

    try:
        if use_logging:
            time_start(f"Downsampling of {target_view.global_offset}")

        num_channels = target_view.header.num_channels
        shape = (num_channels,) + tuple(target_view.size)
        file_buffer = np.zeros(shape, target_view.get_dtype())

        tiles = product(
            *list(
                [
                    list(range(0, math.ceil(len)))
                    for len in np.array(target_view.size) / buffer_edge_len
                ]
            )
        )

        source_view.open()

        for tile in tiles:
            target_offset = np.array(tile) * buffer_edge_len
            source_offset = (mag_factors * target_offset).astype(int)
            source_size = cast(
                Tuple[int, int, int],
                tuple(
                    [
                        int(min(a, b))
                        for a, b in zip(
                            np.array(mag_factors) * buffer_edge_len,
                            source_view.size - source_offset,
                        )
                    ]
                ),
            )

            cube_buffer_channels = source_view.read(source_offset, source_size)

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    # Downsample the buffer
                    data_cube = downsample_cube(
                        cube_buffer,
                        mag_factors,
                        interpolation_mode,
                    )

                    buffer_offset = target_offset
                    buffer_end = buffer_offset + data_cube.shape

                    file_buffer[
                        channel_index,
                        buffer_offset[0] : buffer_end[0],
                        buffer_offset[1] : buffer_end[1],
                        buffer_offset[2] : buffer_end[2],
                    ] = data_cube

        source_view.close()
        # Write the downsampled buffer to target
        if source_view.header.num_channels == 1:
            file_buffer = file_buffer[0]  # remove channel dimension
        target_view.write(file_buffer)
        if use_logging:
            time_stop(f"Downsampling of {target_view.global_offset}")

    except Exception as exc:
        logging.error(f"Downsampling of {target_view.global_offset} failed with {exc}")
        raise exc

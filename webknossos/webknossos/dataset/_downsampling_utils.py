import logging
import math
import warnings
from collections.abc import Callable
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.ndimage import zoom

if TYPE_CHECKING:
    from .dataset import Dataset

from ..geometry import Mag, Vec3Int, Vec3IntLike
from ._array import ArrayInfo
from .layer_categories import LayerCategoryType
from .view import View

logger = logging.getLogger(__name__)


class InterpolationModes(Enum):
    MEDIAN = 0
    MODE = 1
    NEAREST = 2
    BILINEAR = 3
    BICUBIC = 4
    MAX = 5
    MIN = 6


def determine_downsample_buffer_shape(array_info: ArrayInfo) -> Vec3Int:
    # This is the shape of the data in the downsampling target magnification, so the
    # data that is read is up to 512³ vx in the source magnification. Using larger
    # shapes uses a lot of RAM, especially for segmentation layers which use the mode filter.
    # See https://scm.slack.com/archives/CMBMU5684/p1749771929954699 for more context.
    return Vec3Int.full(256).pairmin(array_info.shard_shape)


def determine_upsample_buffer_shape(array_info: ArrayInfo) -> Vec3Int:
    # This is the shape of the data in the upsampling target magnification.
    return array_info.shard_shape


def calculate_mags_to_downsample(
    from_mag: Mag,
    coarsest_mag: Mag,
    dataset_to_align_with: Optional["Dataset"],
    voxel_size: tuple[float, float, float] | None,
) -> list[Mag]:
    assert np.all(from_mag.to_np() <= coarsest_mag.to_np())
    mags = []
    current_mag = from_mag
    if dataset_to_align_with is None:
        mags_to_align_with = set()
    else:
        mags_to_align_with = set(
            mag
            for layer in dataset_to_align_with.layers.values()
            for mag in layer.mags.keys()
        )
    mags_to_align_with_by_max_dim = {mag.max_dim: mag for mag in mags_to_align_with}
    assert len(mags_to_align_with) == len(mags_to_align_with_by_max_dim), (
        "Some layers contain different values for the same mag, this is not allowed."
    )
    while current_mag < coarsest_mag:
        if current_mag.max_dim * 2 in mags_to_align_with_by_max_dim:
            current_mag = mags_to_align_with_by_max_dim[current_mag.max_dim * 2]
            if current_mag > coarsest_mag:
                warnings.warn(
                    "[INFO] The mag taken from another layer is larger in some dimensions than `coarsest_mag`."
                )
        elif voxel_size is None:
            # In case the sampling mode is CONSTANT_Z or ISOTROPIC:
            current_mag = Mag(np.minimum(current_mag.to_np() * 2, coarsest_mag.to_np()))
        else:
            # In case the sampling mode is ANISOTROPIC:
            current_size = current_mag.to_np() * np.array(voxel_size)
            min_value = np.min(current_size)
            min_value_bitmask = np.array(current_size == min_value)
            factor = min_value_bitmask + 1

            # Calculate the two potential magnifications.
            # Either, double all components or only double the smallest component.
            all_voxel_sized = current_size * 2
            min_voxel_sized = (
                current_size * factor
            )  # only multiply the smallest dimension

            all_voxel_sized_ratio = np.max(all_voxel_sized) / np.min(all_voxel_sized)
            min_voxel_sized_ratio = np.max(min_voxel_sized) / np.min(min_voxel_sized)

            # The smaller the ratio between the smallest dimension and the largest dimension, the better.
            if all_voxel_sized_ratio < min_voxel_sized_ratio:
                # Multiply all dimensions with "2"
                new_mag = Mag(np.minimum(current_mag.to_np() * 2, coarsest_mag.to_np()))
            else:
                # Multiply only the minimal dimension by "2".
                new_mag = Mag(
                    np.minimum(current_mag.to_np() * factor, coarsest_mag.to_np())
                )
                # In case of isotropic resolution but anisotropic mag we need to ensure unique max dims.
                # current mag: 4-4-2, voxel_size: 1,1,1 -> new_mag: 4-4-4, therefore we skip this entry.
                if new_mag.max_dim == current_mag.max_dim and new_mag != current_mag:
                    current_mag = new_mag
                    continue
            if new_mag == current_mag:
                raise RuntimeError(
                    f"The coarsest mag {coarsest_mag} can not be reached from {current_mag} with voxel_size {voxel_size}!"
                )
            current_mag = new_mag

        mags += [current_mag]

    return mags


def calculate_mags_to_upsample(
    from_mag: Mag,
    finest_mag: Mag,
    dataset_to_align_with: Optional["Dataset"],
    voxel_size: tuple[float, float, float] | None,
) -> list[Mag]:
    return list(
        reversed(
            calculate_mags_to_downsample(
                finest_mag, from_mag, dataset_to_align_with, voxel_size
            )
        )
    )[1:] + [finest_mag]


def calculate_default_coarsest_mag(dataset_size: Vec3IntLike) -> Mag:
    dataset_size = Vec3Int(dataset_size)
    # The coarsest mag should have a size of ~ 100vx**2 per slice
    coarsest_x_y = max(dataset_size[0], dataset_size[1])
    # highest power of 2 larger (or equal) than coarsest_x_y divided by 100
    # The calculated factor will be used for x, y and z here. If anisotropic downsampling takes place,
    # the dimensions can still be downsampled independently according to the voxel_size.
    return Mag(max(2 ** math.ceil(math.log(coarsest_x_y / 100, 2)), 4))  # at least 4


def parse_interpolation_mode(
    interpolation_mode: str, layer_category: LayerCategoryType
) -> InterpolationModes:
    if interpolation_mode.upper() == "DEFAULT":
        return (
            InterpolationModes.MODE
            if layer_category == "segmentation"
            else InterpolationModes.MEDIAN
        )
    else:
        return InterpolationModes[interpolation_mode.upper()]


def linear_filter_3d(data: np.ndarray, factors: list[int], order: int) -> np.ndarray:
    factors_np = np.array(factors)

    return zoom(
        data,
        1 / factors_np,
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
    data: np.ndarray, factors: list[int], func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    ds = data.shape
    assert not any(d % factor > 0 for (d, factor) in zip(ds, factors))
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
    index = list(np.ogrid[slices])
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[tuple(index)]


def downsample_unpadded_data(
    buffer: np.ndarray, target_mag: Mag, interpolation_mode: InterpolationModes
) -> np.ndarray:
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
    cube_buffer: np.ndarray, factors: list[int], interpolation_mode: InterpolationModes
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
        raise Exception(f"Invalid interpolation mode: {interpolation_mode}")


def downsample_cube_job(
    args: tuple[View, View, int],
    mag_factors: Vec3Int,
    interpolation_mode: InterpolationModes,
    buffer_shape: Vec3Int,
) -> None:
    (source_view, target_view, _i) = args

    try:
        num_channels = target_view.info.num_channels
        target_bbox_in_mag = target_view.bounding_box.in_mag(target_view.mag)
        shape = (num_channels,) + target_bbox_in_mag.size.to_tuple()
        shape_xyz = target_bbox_in_mag.size_xyz
        file_buffer = np.zeros(shape, target_view.get_dtype())

        tiles = product(
            *(
                list(range(0, math.ceil(length / buffer_edge_len)))
                for length, buffer_edge_len in zip(shape_xyz, buffer_shape)
            )
        )

        for tile in tiles:
            target_offset = Vec3Int(tile) * buffer_shape
            source_offset = target_offset * target_view.mag
            source_size = source_view.bounding_box.size_xyz
            source_size = (buffer_shape * target_view.mag).pairmin(
                source_size - source_offset
            )

            bbox = source_view.bounding_box.offset(source_offset).with_size_xyz(
                source_size
            )

            cube_buffer_channels = source_view.read_xyz(
                absolute_bounding_box=bbox,
            )

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    # Downsample the buffer
                    data_cube = downsample_cube(
                        cube_buffer,
                        mag_factors.to_list(),
                        interpolation_mode,
                    )

                    buffer_bbox = target_view.bounding_box.with_topleft_xyz(
                        target_offset
                    ).with_size_xyz(data_cube.shape)

                    # Add missing axes to the data_cube if bbox is nd
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)

                    file_buffer[(channel_index,) + buffer_bbox.to_slices_xyz()] = (
                        data_cube
                    )

        # Write the downsampled buffer to target
        target_view.write(file_buffer, absolute_bounding_box=target_view.bounding_box)

    except Exception as exc:
        logger.error(
            f"Downsampling of target {target_view.bounding_box} failed with {exc}"
        )
        raise exc

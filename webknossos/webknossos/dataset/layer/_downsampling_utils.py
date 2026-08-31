import logging
import math
import warnings
from collections.abc import Callable
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Union

import numba
import numpy as np
from scipy.ndimage import zoom

if TYPE_CHECKING:
    from webknossos.dataset.dataset import Dataset, RemoteDataset

from webknossos.dataset_properties import LayerCategoryType
from webknossos.geometry import Mag, Vec3FloatLike, Vec3Int, Vec3IntLike

from .view import ArrayInfo, View

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
    # data that is read is up to 1024³ vx in the source magnification. Using larger
    # shapes uses a lot of RAM, especially for segmentation layers which use the mode filter.
    # See https://scm.slack.com/archives/CMBMU5684/p1749771929954699 for more context.
    return Vec3Int.full(512).pairmin(array_info.shard_shape)


def determine_upsample_buffer_shape(array_info: ArrayInfo) -> Vec3Int:
    # This is the shape of the data in the upsampling target magnification.
    return array_info.shard_shape


def calculate_mags_to_downsample(
    from_mag: Mag,
    coarsest_mag: Mag,
    dataset_to_align_with: Union["Dataset", "RemoteDataset"] | None,
    voxel_size: Vec3FloatLike | None,
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
    dataset_to_align_with: Union["Dataset", "RemoteDataset"] | None,
    voxel_size: Vec3FloatLike | None,
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


@numba.jit(nopython=True, nogil=True)
def _mode(input_array: np.ndarray) -> np.ndarray:
    values = np.zeros(input_array.shape[0], dtype=input_array.dtype)
    counter = np.zeros(input_array.shape[0], dtype=np.uint8)
    output_array = np.zeros(input_array.shape[1], dtype=input_array.dtype)
    for row_index in range(input_array.shape[1]):
        values[0] = input_array[0, row_index]
        counter[:] = 1
        value_offset = 1
        for col_index in range(1, input_array.shape[0]):
            value = input_array[col_index, row_index]
            found_value = False
            for i in range(
                value_offset
            ):  # Only iterate the values that were already seen
                if value == values[i]:
                    counter[i] = counter[i] + 1
                    found_value = True
                    break
            if not found_value:
                values[value_offset] = value
                value_offset += 1
        mode = values[np.argmax(counter)]
        output_array[row_index] = mode

    return output_array


# These kernels read the source buffer directly, which avoids the three full copies
# that reshaping it into an (elements-per-block, num-blocks) array costs.
# Within a block the elements are visited in the order `dy + fy * dx + fx * fy * dz`,
# which is the order `non_linear_filter_3d` produces and which determines how the
# mode filter breaks ties.


@numba.jit(nopython=True, nogil=True)
def _median_kernel(
    source: np.ndarray, target: np.ndarray, fx: int, fy: int, fz: int
) -> None:
    n = fx * fy * fz
    half = n // 2
    block = np.empty(n, dtype=source.dtype)
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                i = 0
                for dz in range(fz):
                    for dy in range(fy):
                        for dx in range(fx):
                            block[i] = source[x * fx + dx, y * fy + dy, z * fz + dz]
                            i += 1
                # Insertion sort, n is tiny (8 for an isotropic step).
                for a in range(1, n):
                    value = block[a]
                    b = a - 1
                    while b >= 0 and block[b] > value:
                        block[b + 1] = block[b]
                        b -= 1
                    block[b + 1] = value
                if n % 2 == 1:
                    target[x, y, z] = block[half]
                else:
                    # np.median averages the two middle elements in float64 and
                    # truncates to the input dtype.
                    target[x, y, z] = (
                        np.float64(block[half - 1]) + np.float64(block[half])
                    ) / 2.0


@numba.jit(nopython=True, nogil=True)
def _median_kernel_2x2x2(source: np.ndarray, target: np.ndarray) -> None:
    # Batcher odd-even merge network for 8 elements, reduced to the comparators
    # needed to place the 4th and 5th smallest element.
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                a0 = source[2 * x, 2 * y, 2 * z]
                a1 = source[2 * x + 1, 2 * y, 2 * z]
                a2 = source[2 * x, 2 * y + 1, 2 * z]
                a3 = source[2 * x + 1, 2 * y + 1, 2 * z]
                a4 = source[2 * x, 2 * y, 2 * z + 1]
                a5 = source[2 * x + 1, 2 * y, 2 * z + 1]
                a6 = source[2 * x, 2 * y + 1, 2 * z + 1]
                a7 = source[2 * x + 1, 2 * y + 1, 2 * z + 1]

                b0 = min(a0, a1)
                b1 = max(a0, a1)
                b2 = min(a2, a3)
                b3 = max(a2, a3)
                b4 = min(a4, a5)
                b5 = max(a4, a5)
                b6 = min(a6, a7)
                b7 = max(a6, a7)

                c0 = min(b0, b2)
                c1 = min(b1, b3)
                c2 = max(b0, b2)
                c3 = max(b1, b3)
                c4 = min(b4, b6)
                c5 = min(b5, b7)
                c6 = max(b4, b6)
                c7 = max(b5, b7)

                d1 = min(c1, c2)
                d2 = max(c1, c2)
                d5 = min(c5, c6)
                d6 = max(c5, c6)

                e2 = min(d2, d6)
                e3 = min(c3, c7)
                e4 = max(c0, c4)
                e5 = max(d1, d5)

                f3 = min(e3, e5)
                f4 = max(e2, e4)

                target[x, y, z] = (
                    np.float64(min(f3, f4)) + np.float64(max(f3, f4))
                ) / 2.0


@numba.jit(nopython=True, nogil=True)
def _mode_kernel(
    source: np.ndarray, target: np.ndarray, fx: int, fy: int, fz: int
) -> None:
    n = fx * fy * fz
    values = np.empty(n, dtype=source.dtype)
    counts = np.empty(n, dtype=np.int64)
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                num_values = 0
                for dz in range(fz):
                    for dx in range(fx):
                        for dy in range(fy):
                            value = source[x * fx + dx, y * fy + dy, z * fz + dz]
                            found_value = False
                            for i in range(num_values):
                                if values[i] == value:
                                    counts[i] += 1
                                    found_value = True
                                    break
                            if not found_value:
                                values[num_values] = value
                                counts[num_values] = 1
                                num_values += 1
                # Ties are won by the value that occurs first in the block.
                best = 0
                for i in range(1, num_values):
                    if counts[i] > counts[best]:
                        best = i
                target[x, y, z] = values[best]


@numba.jit(nopython=True, nogil=True)
def _mode_kernel_2x2x2(source: np.ndarray, target: np.ndarray) -> None:
    # Each element counts only the matches that follow it, so the first occurrence
    # of a value carries its full count and ties are won by the earliest element.
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                a0 = source[2 * x, 2 * y, 2 * z]
                a1 = source[2 * x, 2 * y + 1, 2 * z]
                a2 = source[2 * x + 1, 2 * y, 2 * z]
                a3 = source[2 * x + 1, 2 * y + 1, 2 * z]
                a4 = source[2 * x, 2 * y, 2 * z + 1]
                a5 = source[2 * x, 2 * y + 1, 2 * z + 1]
                a6 = source[2 * x + 1, 2 * y, 2 * z + 1]
                a7 = source[2 * x + 1, 2 * y + 1, 2 * z + 1]

                c0 = (
                    1
                    + (a0 == a1)
                    + (a0 == a2)
                    + (a0 == a3)
                    + (a0 == a4)
                    + (a0 == a5)
                    + (a0 == a6)
                    + (a0 == a7)
                )
                c1 = (
                    1
                    + (a1 == a2)
                    + (a1 == a3)
                    + (a1 == a4)
                    + (a1 == a5)
                    + (a1 == a6)
                    + (a1 == a7)
                )
                c2 = 1 + (a2 == a3) + (a2 == a4) + (a2 == a5) + (a2 == a6) + (a2 == a7)
                c3 = 1 + (a3 == a4) + (a3 == a5) + (a3 == a6) + (a3 == a7)
                c4 = 1 + (a4 == a5) + (a4 == a6) + (a4 == a7)
                c5 = 1 + (a5 == a6) + (a5 == a7)
                c6 = 1 + (a6 == a7)

                best = a0
                best_count = c0
                if c1 > best_count:
                    best = a1
                    best_count = c1
                if c2 > best_count:
                    best = a2
                    best_count = c2
                if c3 > best_count:
                    best = a3
                    best_count = c3
                if c4 > best_count:
                    best = a4
                    best_count = c4
                if c5 > best_count:
                    best = a5
                    best_count = c5
                if c6 > best_count:
                    best = a6
                target[x, y, z] = best


def _can_use_kernel(data: np.ndarray, factors: list[int]) -> bool:
    # Floating point data uses the numpy implementation, because np.median
    # accumulates in the input dtype for float32 and has its own NaN semantics.
    return data.ndim == 3 and len(factors) == 3 and data.dtype.kind in "iu"


def _apply_kernel(
    data: np.ndarray,
    factors: list[int],
    kernel: Callable[[np.ndarray, np.ndarray, int, int, int], None],
    kernel_2x2x2: Callable[[np.ndarray, np.ndarray], None],
) -> np.ndarray:
    assert not any(d % factor > 0 for (d, factor) in zip(data.shape, factors))
    fx, fy, fz = factors
    target = np.empty(
        (data.shape[0] // fx, data.shape[1] // fy, data.shape[2] // fz),
        dtype=data.dtype,
        order="F",
    )
    if fx == 2 and fy == 2 and fz == 2:
        kernel_2x2x2(data, target)
    else:
        kernel(data, target, fx, fy, fz)
    return target


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
        if _can_use_kernel(cube_buffer, factors):
            return _apply_kernel(cube_buffer, factors, _mode_kernel, _mode_kernel_2x2x2)
        return non_linear_filter_3d(cube_buffer, factors, _mode)
    elif interpolation_mode == InterpolationModes.MEDIAN:
        if _can_use_kernel(cube_buffer, factors):
            return _apply_kernel(
                cube_buffer, factors, _median_kernel, _median_kernel_2x2x2
            )
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
        source_bbox = source_view.normalized_bounding_box
        target_bbox = target_view.normalized_bounding_box
        num_channels = target_view.info.bounding_box.size.c
        target_bbox_in_mag = target_bbox.in_mag(target_view.mag)
        shape = target_bbox_in_mag.size.to_tuple()
        shape_xyz = target_bbox_in_mag.size_xyz
        file_buffer = np.zeros(shape, target_view.get_dtype(), order="F")

        tiles = product(
            *(
                list(range(0, math.ceil(length / buffer_edge_len)))
                for length, buffer_edge_len in zip(shape_xyz, buffer_shape)
            )
        )

        for tile in tiles:
            target_offset = Vec3Int(tile) * buffer_shape
            source_offset = target_offset * target_view.mag
            source_size = source_bbox.size_xyz
            source_size = (buffer_shape * target_view.mag).pairmin(
                source_size - source_offset
            )

            bbox = source_bbox.offset(source_offset).with_size_xyz(source_size)

            cube_buffer_channels = source_view.read_cxyz(
                absolute_bounding_box=bbox,
            )

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if cube_buffer.any():
                    # Downsample the buffer
                    data_cube = downsample_cube(
                        cube_buffer,
                        mag_factors.to_list(),
                        interpolation_mode,
                    )

                    buffer_bbox = target_bbox.with_topleft_xyz(
                        target_offset
                    ).with_size_xyz(data_cube.shape)
                    if "c" in buffer_bbox.axes:
                        buffer_bbox = buffer_bbox.with_bounds("c", new_size=1)

                    # Add missing axes to the data_cube if bbox is nd
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)

                    slices: list[int | slice] = list(buffer_bbox.to_slices_xyz())
                    if "c" in buffer_bbox.axes:
                        slices[buffer_bbox.index.c] = channel_index

                    file_buffer[tuple(slices)] = data_cube

        # Write the downsampled buffer to target
        target_view.write(file_buffer, absolute_bounding_box=target_bbox)

    except Exception as exc:
        logger.exception(f"Downsampling of target {target_bbox} failed with {exc}")
        raise exc

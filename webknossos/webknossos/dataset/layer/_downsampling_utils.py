import logging
import math
import warnings
from collections.abc import Callable, Iterator
from enum import Enum
from itertools import product
from os import environ
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.ndimage import zoom

if TYPE_CHECKING:
    from webknossos.dataset.dataset import Dataset, RemoteDataset

from webknossos.dataset_properties import LayerCategoryType
from webknossos.geometry import Mag, Vec3FloatLike, Vec3Int, Vec3IntLike

from .view import ArrayInfo, View

try:
    from . import _downsampling_numba
except ImportError:
    # numba is an optional dependency, the numpy implementations below are used instead.
    _downsampling_numba = None  # type: ignore[assignment]

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


# Numpy fallbacks for the non-linear filters, used when numba is not installed.
# Within a block the elements are ordered `dy + fy * dx + fx * fy * dz`, which is the
# order `non_linear_filter_3d` produces and which determines how the mode filter
# breaks ties. The work is done in slabs to bound the size of the temporaries.

_SLAB_VOXELS = 4 * 1024 * 1024

# Batcher odd-even merge network for 8 elements, reduced to the comparators needed to
# place the 4th and 5th smallest element.
_MEDIAN_NETWORK_8 = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (1, 2), (5, 6),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (2, 4), (3, 5),
    (3, 4),
]  # fmt: skip


def _block_views(data: np.ndarray, factors: list[int]) -> list[np.ndarray]:
    fx, fy, fz = factors
    return [
        data[dx::fx, dy::fy, dz::fz]
        for dz in range(fz)
        for dx in range(fx)
        for dy in range(fy)
    ]


def _empty_target(data: np.ndarray, factors: list[int]) -> np.ndarray:
    assert not any(d % factor > 0 for (d, factor) in zip(data.shape, factors))
    fx, fy, fz = factors
    return np.empty(
        (data.shape[0] // fx, data.shape[1] // fy, data.shape[2] // fz),
        dtype=data.dtype,
        order="F",
    )


def _slabs(target: np.ndarray) -> Iterator[tuple[int, int]]:
    plane_voxels = max(target.shape[0] * target.shape[1], 1)
    planes = max(_SLAB_VOXELS // plane_voxels, 1)
    for z in range(0, target.shape[2], planes):
        yield z, min(z + planes, target.shape[2])


def _median_numpy_2x2x2(data: np.ndarray) -> np.ndarray:
    target = _empty_target(data, [2, 2, 2])
    views = _block_views(data, [2, 2, 2])
    for z_start, z_stop in _slabs(target):
        block = [np.array(view[:, :, z_start:z_stop]) for view in views]
        scratch = np.empty_like(block[0])
        for i, j in _MEDIAN_NETWORK_8:
            np.minimum(block[i], block[j], out=scratch)
            np.maximum(block[i], block[j], out=block[j])
            block[i], scratch = scratch, block[i]
        # np.median averages the two middle elements in float64 and truncates.
        middle = (block[3].astype(np.float64) + block[4].astype(np.float64)) / 2.0
        target[:, :, z_start:z_stop] = middle.astype(data.dtype)
    return target


def _mode_numpy(data: np.ndarray, factors: list[int]) -> np.ndarray:
    target = _empty_target(data, factors)
    views = _block_views(data, factors)
    num_elements = len(views)
    for z_start, z_stop in _slabs(target):
        block = [view[:, :, z_start:z_stop] for view in views]
        # Each element counts only the matches that follow it, so the first occurrence
        # of a value carries its full count and ties are won by the earliest element.
        best = np.array(block[0])
        best_count = np.ones(best.shape, dtype=np.uint16)
        for j in range(1, num_elements):
            best_count += block[0] == block[j]
        count = np.empty_like(best_count)
        # The last element always has a count of 1 and can never win.
        for i in range(1, num_elements - 1):
            count[...] = 1
            for j in range(i + 1, num_elements):
                count += block[i] == block[j]
            wins = count > best_count
            np.copyto(best, block[i], where=wins)
            np.copyto(best_count, count, where=wins)
        target[:, :, z_start:z_stop] = best
    return target


def _supports_fast_path(data: np.ndarray, factors: list[int]) -> bool:
    return data.ndim == 3 and len(factors) == 3


def _warn_once_about_missing_numba() -> None:
    # The environment is inherited by worker processes, so they stay quiet as well.
    if environ.get("WEBKNOSSOS_SHOWED_MISSING_NUMBA_WARNING", "False") == "False":
        environ["WEBKNOSSOS_SHOWED_MISSING_NUMBA_WARNING"] = "True"
        warnings.warn(
            "[INFO] Downsampling uses the numpy implementation of the non-linear "
            "filters. Install the optional numba dependency with "
            "'pip install webknossos[numba]' for a faster implementation.",
            category=UserWarning,
            stacklevel=2,
        )


def warn_if_numba_is_missing(interpolation_mode: InterpolationModes) -> None:
    """Emits the warning in this process, so that worker processes started afterwards
    inherit the environment flag and stay quiet."""
    if _downsampling_numba is None and interpolation_mode in (
        InterpolationModes.MEDIAN,
        InterpolationModes.MODE,
    ):
        _warn_once_about_missing_numba()


def _downsample_median(data: np.ndarray, factors: list[int]) -> np.ndarray:
    # Floating point data uses the numpy implementation, because np.median
    # accumulates in the input dtype for float32 and has its own NaN semantics.
    if _supports_fast_path(data, factors) and data.dtype.kind in "iu":
        if _downsampling_numba is not None:
            return _downsampling_numba.median_downsample(data, factors)
        _warn_once_about_missing_numba()
        if factors == [2, 2, 2]:
            return _median_numpy_2x2x2(data)
    return non_linear_filter_3d(data, factors, _median)


def _downsample_mode(data: np.ndarray, factors: list[int]) -> np.ndarray:
    if _downsampling_numba is not None and _supports_fast_path(data, factors):
        return _downsampling_numba.mode_downsample(data, factors)
    _warn_once_about_missing_numba()
    return _mode_numpy(data, factors)


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
        return _downsample_mode(cube_buffer, factors)
    elif interpolation_mode == InterpolationModes.MEDIAN:
        return _downsample_median(cube_buffer, factors)
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

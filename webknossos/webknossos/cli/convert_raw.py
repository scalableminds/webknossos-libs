"""This module converts a RAW dataset to a WEBKNOSSOS dataset."""

import logging
from argparse import Namespace
from functools import partial
from multiprocessing import cpu_count
from typing import Annotated, Any, Literal

import numpy as np
import typer
from upath import UPath

from ..dataset import Dataset, MagView, SamplingModes
from ..dataset.defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from ..dataset_properties import DataFormat, LengthUnit, VoxelSize
from ..dataset_properties.structuring import DEFAULT_LENGTH_UNIT_STR
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import (
    get_executor_for_args,
    is_fs_path,
    rmtree,
    time_start,
    time_stop,
    wait_and_ensure_success,
)
from ._utils import (
    DistributionStrategy,
    Order,
    RescaleValues,
    SamplingMode,
    VoxelSizeTuple,
    parse_mag,
    parse_path,
    parse_rescale_values,
    parse_vec3int,
    parse_voxel_size,
    prepare_shard_shape,
)

logger = logging.getLogger(__name__)


def _raw_chunk_converter(
    bounding_box: BoundingBox,
    source_raw_path: UPath,
    target_mag_view: MagView,
    source_dtype: np.dtype,
    target_dtype: np.dtype,
    shape: Vec3Int,
    order: Literal["C", "F"],
    flip_axes: tuple[int, ...] | None,
    rescale_min_max: RescaleValues | None,
) -> None:
    logger.info("Conversion of %s", bounding_box.topleft)
    assert is_fs_path(source_raw_path)
    source_data: np.ndarray = np.memmap(
        str(
            source_raw_path
        ),  # this is fine, because we checked that source_raw_path is a fs_path
        dtype=source_dtype,
        mode="r",
        shape=(1,) + shape.to_tuple(),
        order=order,
    )

    if flip_axes:
        source_data = np.flip(source_data, flip_axes)

    contiguous_chunk = source_data[(slice(None),) + bounding_box.to_slices()].copy(
        order="F"
    )
    if rescale_min_max is not None:
        if np.isclose(rescale_min_max.min, rescale_min_max.max):
            contiguous_chunk = np.zeros_like(contiguous_chunk, dtype=target_dtype)
        else:
            target_max = (
                np.iinfo(target_dtype).max
                if np.issubdtype(target_dtype, np.integer)
                else 1
            )
            norm = (contiguous_chunk.astype(np.float64) - rescale_min_max.min) / (
                rescale_min_max.max - rescale_min_max.min
            )
            contiguous_chunk = norm * target_max

    if contiguous_chunk.dtype != target_dtype:
        contiguous_chunk = contiguous_chunk.astype(target_dtype)
    target_mag_view.write(data=contiguous_chunk, absolute_offset=bounding_box.topleft)


def convert_raw(
    *,
    source_raw_path: UPath,
    target_path: UPath,
    layer_name: str,
    source_dtype: np.dtype,
    target_dtype: np.dtype,
    shape: Vec3Int,
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    shard_shape: Vec3Int,
    order: Literal["C", "F"] = "F",
    voxel_size_with_unit: VoxelSize = VoxelSize((1.0, 1.0, 1.0)),
    flip_axes: tuple[int, ...] | None = None,
    compress: bool = True,
    rescale_min_max: RescaleValues | None = None,
    executor_args: Namespace | None = None,
) -> MagView:
    """Performs the conversion step from RAW file to WEBKNOSSOS"""
    time_start(f"Conversion of {source_raw_path}")

    wk_ds = Dataset(
        target_path, voxel_size_with_unit=voxel_size_with_unit, exist_ok=True
    )
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "color",
        dtype=np.dtype(target_dtype),
        num_channels=1,
        data_format=data_format,
    )
    wk_layer.bounding_box = BoundingBox((0, 0, 0), shape)
    wk_mag = wk_layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=compress,
    )

    # Parallel chunk conversion
    with get_executor_for_args(args=executor_args) as executor:
        wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    _raw_chunk_converter,
                    source_raw_path=source_raw_path,
                    target_mag_view=wk_mag,
                    source_dtype=source_dtype,
                    target_dtype=target_dtype,
                    shape=shape,
                    order=order,
                    flip_axes=flip_axes,
                    rescale_min_max=rescale_min_max,
                ),
                wk_layer.bounding_box.chunk(chunk_shape=shard_shape),
            ),
            executor=executor,
        )

    time_stop(f"Conversion of {source_raw_path}")
    return wk_mag


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your raw image data.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Target path to save your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Shape of the source dataset. Should be a comma separated "
            "string (e.g. `1024,1024,512`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ],
    voxel_size: Annotated[
        VoxelSizeTuple,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma-separated string (e.g. `11.0,11.0,20.0`).",
            parser=parse_voxel_size,
            metavar="VoxelSize",
            show_default=False,
        ),
    ],
    unit: Annotated[
        LengthUnit,
        typer.Option(
            help="The unit of the voxel size.",
        ),
    ] = DEFAULT_LENGTH_UNIT_STR,  # type:ignore
    dtype: Annotated[
        str, typer.Option(help="Target datatype (e.g. `uint8`, `uint16`, `uint32`)")
    ] = "uint8",
    source_dtype: Annotated[
        str | None,
        typer.Option(
            help="Source datatype (e.g. `uint8`, `uint16`, `uint32`). "
            "If omitted, it is assumed to be the same as the target datatype."
        ),
    ] = None,
    order: Annotated[
        Order,
        typer.Option(
            help="The input data storage layout: "
            "either 'F' for Fortran-style/column-major order (the default), "
            "or 'C' for C-style/row-major order. "
            "Note: Axes are expected in (x, y, z) order."
        ),
    ] = Order.F,
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the output layer (color or segmentation)"),
    ] = "color",
    rescale_min_max: Annotated[
        RescaleValues | None,
        typer.Option(
            help="Rescale the values of the target dataset by specifying the min and max values. "
            "Will be scaled to the range from 0 to the maximum value of the target data type or 1.0 for floats. "
            "Should be a comma-separated string (e.g. `0.2,0.8`).",
            parser=parse_rescale_values,
            metavar="RescaleValues",
            show_default=False,
        ),
    ] = None,
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = str(DEFAULT_DATA_FORMAT),  # type:ignore
    chunk_shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    shard_shape: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Number of voxels to be stored as a shard in the output format "
            "(e.g. `1024` or `1024,1024,1024`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    chunks_per_shard: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Deprecated, use --shard-shape. Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    flip_axes: Annotated[
        Vec3Int | None,
        typer.Option(
            help="The axes that should be flipped. "
            "Input format is a comma-separated list of axis indices. "
            "For example, 1,2,3 will flip the x, y and z axes.",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
    ] = True,
    downsample: Annotated[
        bool, typer.Option(help="Downsample the target dataset.")
    ] = True,
    max_mag: Annotated[
        Mag | None,
        typer.Option(
            help="Create downsampled magnifications up to the magnification specified by this argument. "
            "If omitted, the coarsest magnification will be determined by using the bounding box of the layer. "
            "Should be number or hyphen-separated string (e.g. `2` or `2-2-2`).",
            parser=parse_mag,
        ),
    ] = None,
    interpolation_mode: Annotated[
        str,
        typer.Option(
            help="The interpolation mode that should be used "
            "(median, mode, nearest, bilinear or bicubic)."
        ),
    ] = "default",
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    overwrite_existing: Annotated[
        bool,
        typer.Option(
            help="Clear target folder if it already exists. Not enabled by default. Use with caution.",
            show_default=False,
        ),
    ] = False,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        str | None,
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string \
(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Converts a RAW file into a WEBKNOSSOS dataset."""

    if source_dtype is None:
        source_dtype = dtype

    shard_shape = prepare_shard_shape(
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        chunks_per_shard=chunks_per_shard,
    )

    if flip_axes is not None:
        for index in flip_axes:
            assert 0 <= index <= 3, (
                "flip_axes parameter must only contain indices between 0 and 3."
            )

    mode = SamplingModes.parse(sampling_mode.value)
    if source.is_dir():
        logger.error("source_path is not a file")
        return

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    voxel_size_with_unit = VoxelSize(voxel_size, unit)

    if overwrite_existing and target.exists():
        rmtree(target)

    mag_view = convert_raw(
        source_raw_path=source,
        target_path=target,
        layer_name=layer_name,
        source_dtype=np.dtype(source_dtype),
        target_dtype=np.dtype(dtype),
        shape=shape,
        data_format=data_format,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape or DEFAULT_SHARD_SHAPE,
        order=order.value,
        voxel_size_with_unit=voxel_size_with_unit,
        flip_axes=flip_axes.to_tuple() if flip_axes else None,
        compress=compress,
        rescale_min_max=rescale_min_max,
        executor_args=executor_args,
    )

    if downsample:
        with get_executor_for_args(executor_args) as executor:
            mag_view.layer.downsample(
                from_mag=mag_view.mag,
                coarsest_mag=max_mag,
                interpolation_mode=interpolation_mode,
                compress=compress,
                sampling_mode=mode,
                executor=executor,
            )

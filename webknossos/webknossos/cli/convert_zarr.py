"""This module converts a Zarr dataset to a WEBKNOSSOS dataset."""

import logging
import time
from argparse import Namespace
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import tensorstore
import typer

from webknossos.dataset.length_unit import LengthUnit
from webknossos.dataset.properties import DEFAULT_LENGTH_UNIT_STR, VoxelSize

from ..dataset import DataFormat, Dataset, MagView, SegmentationLayer
from ..dataset.defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import get_executor_for_args, wait_and_ensure_success
from ._utils import (
    DistributionStrategy,
    SamplingMode,
    VoxelSizeTuple,
    parse_mag,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
    prepare_shard_shape,
)

logger = logging.getLogger(__name__)


def _try_open_zarr(path: Path) -> tensorstore.TensorStore:
    try:
        return tensorstore.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}
        ).result()
    except tensorstore.TensorStoreError:
        return tensorstore.open(
            {"driver": "zarr", "kvstore": {"driver": "file", "path": path}}
        ).result()


def _zarr_chunk_converter(
    bounding_box: BoundingBox,
    source_zarr_path: Path,
    target_mag_view: MagView,
    flip_axes: int | tuple[int, ...] | None,
) -> int:
    logger.info("Conversion of %s", bounding_box.topleft)

    slices = bounding_box.to_slices()
    zarr_file = _try_open_zarr(source_zarr_path)
    source_data: np.ndarray = zarr_file[slices].read().result()[None, ...]

    if flip_axes:
        source_data = np.flip(source_data, flip_axes)

    contiguous_chunk = source_data.copy(order="F")
    target_mag_view.write(data=contiguous_chunk, absolute_offset=bounding_box.topleft)

    return source_data.max()


def convert_zarr(
    source_zarr_path: Path,
    target_path: Path,
    layer_name: str,
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    shard_shape: Vec3Int,
    is_segmentation_layer: bool = False,
    voxel_size_with_unit: VoxelSize = VoxelSize((1.0, 1.0, 1.0)),
    flip_axes: int | tuple[int, ...] | None = None,
    compress: bool = True,
    executor_args: Namespace | None = None,
) -> MagView:
    """Performs the conversation of a Zarr dataset to a WEBKNOSSOS dataset."""
    ref_time = time.time()

    file = _try_open_zarr(source_zarr_path)
    input_dtype: np.dtype = file.dtype
    shape: tuple[int, ...] = file.domain.exclusive_max

    wk_ds = Dataset(
        target_path, voxel_size_with_unit=voxel_size_with_unit, exist_ok=True
    )
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "segmentation" if is_segmentation_layer else "color",
        dtype_per_channel=np.dtype(input_dtype),
        num_channels=1,
        largest_segment_id=0,
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
        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(
                partial(
                    _zarr_chunk_converter,
                    source_zarr_path=source_zarr_path,
                    target_mag_view=wk_mag,
                    flip_axes=flip_axes,
                ),
                wk_layer.bounding_box.chunk(chunk_shape=shard_shape),
            ),
            executor=executor,
        )

    if is_segmentation_layer:
        largest_segment_id = int(max(largest_segment_id_per_chunk))
        cast(SegmentationLayer, wk_layer).largest_segment_id = largest_segment_id

    logger.debug(
        "Conversion of %s took %.8fs", source_zarr_path, time.time() - ref_time
    )
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
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    voxel_size: Annotated[
        VoxelSizeTuple,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma separated string (e.g. 11.0,11.0,20.0).",
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
    max_mag: Annotated[
        Mag | None,
        typer.Option(
            help="Max resolution to be downsampled. "
            "Should be number or minus separated string (e.g. 2 or 2-2-2).",
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
    flip_axes: Annotated[
        Vec3Int | None,
        typer.Option(
            help="The axes at which should be flipped. "
            "Input format is a comma separated list of axis indices. "
            "For example, 1,2,3 will flip the x, y and z axes.",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
    ] = False,
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    is_segmentation_layer: Annotated[
        bool,
        typer.Option(
            help="When converting one layer, signals whether layer is segmentation layer. \
When converting a folder, this option is ignored."
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
    """Converts a Zarr dataset to a WEBKNOSSOS dataset."""

    if not source.is_dir():
        logger.error("source_path is not a directory")
        return

    shard_shape = prepare_shard_shape(
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        chunks_per_shard=chunks_per_shard,
    )

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    voxel_size_with_unit = VoxelSize(voxel_size, unit)

    mag_view = convert_zarr(
        source,
        target,
        layer_name=layer_name,
        data_format=data_format,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape or DEFAULT_SHARD_SHAPE,
        is_segmentation_layer=is_segmentation_layer,
        voxel_size_with_unit=voxel_size_with_unit,
        flip_axes=flip_axes,
        compress=compress,
        executor_args=executor_args,
    )

    with get_executor_for_args(executor_args) as executor:
        mag_view.layer.downsample(
            from_mag=mag_view.mag,
            coarsest_mag=max_mag,
            interpolation_mode=interpolation_mode,
            compress=compress,
            sampling_mode=sampling_mode,
            executor=executor,
        )

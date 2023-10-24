"""This module converts a RAW dataset to a WEBKNOSSOS dataset."""

import logging
from argparse import Namespace
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import typer
from typing_extensions import Annotated

from ..dataset import DataFormat, Dataset, MagView, SamplingModes
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE, DEFAULT_CHUNKS_PER_SHARD
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import (
    get_executor_for_args,
    time_start,
    time_stop,
    wait_and_ensure_success,
)
from ._utils import (
    DistributionStrategy,
    Order,
    SamplingMode,
    VoxelSize,
    parse_mag,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
)

logger = logging.getLogger(__name__)


def _raw_chunk_converter(
    bounding_box: BoundingBox,
    source_raw_path: Path,
    target_mag_view: MagView,
    input_dtype: str,
    shape: Tuple[int, int, int],
    order: Literal["C", "F"],
    flip_axes: Optional[Union[int, Tuple[int, ...]]],
) -> None:
    logging.info("Conversion of %s", bounding_box.topleft)
    source_data: np.ndarray = np.memmap(
        source_raw_path,
        dtype=np.dtype(input_dtype),
        mode="r",
        shape=(1,) + shape,
        order=order,
    )

    if flip_axes:
        source_data = np.flip(source_data, flip_axes)

    contiguous_chunk = source_data[(slice(None),) + bounding_box.to_slices()].copy(
        order="F"
    )
    target_mag_view.write(contiguous_chunk, bounding_box.topleft)


def convert_raw(
    source_raw_path: Path,
    target_path: Path,
    layer_name: str,
    input_dtype: str,
    shape: Vec3Int,
    data_format: DataFormat,
    chunk_shape: Vec3Int,
    chunks_per_shard: Vec3Int,
    order: Literal["C", "F"] = "F",
    voxel_size: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    flip_axes: Optional[Union[int, Tuple[int, ...]]] = None,
    compress: bool = True,
    executor_args: Optional[Namespace] = None,
) -> MagView:
    """Performs the conversion step from RAW file to WEBKNOSSOS"""
    time_start(f"Conversion of {source_raw_path}")

    if voxel_size is None:
        voxel_size = 1.0, 1.0, 1.0
    wk_ds = Dataset(target_path, voxel_size=voxel_size, exist_ok=True)
    wk_layer = wk_ds.get_or_add_layer(
        layer_name,
        "color",
        dtype_per_layer=np.dtype(input_dtype),
        num_channels=1,
        data_format=data_format,
    )
    wk_layer.bounding_box = BoundingBox((0, 0, 0), shape)
    wk_mag = wk_layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        chunks_per_shard=chunks_per_shard,
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
                    input_dtype=input_dtype,
                    shape=shape,
                    order=order,
                    flip_axes=flip_axes,
                ),
                wk_layer.bounding_box.chunk(chunk_shape=chunk_shape * chunks_per_shard),
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
            help="Shape of the source dataset. Should be a comma seperated "
            "sting (e.g. 1024,1024,512).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ],
    order: Annotated[
        Order,
        typer.Option(
            help="The input data storage layout: "
            "either 'F' for Fortran-style/column-major order (the default), "
            "or 'C' for C-style/row-major order. "
            "Note: Axes are expected in  (x, y, z) order."
        ),
    ] = Order.F,
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    voxel_size: Annotated[
        Optional[VoxelSize],
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma seperated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VOXEL_SIZE",
        ),
    ] = None,
    dtype: Annotated[
        str, typer.Option(help="Target datatype (e.g. uint8, uint16, uint32)")
    ] = "uint8",
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = "wkw",  # type:ignore
    chunk_shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    chunks_per_shard: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNKS_PER_SHARD,
    max_mag: Annotated[
        Optional[Mag],
        typer.Option(
            help="Max resolution to be downsampled. "
            "Should be number or minus seperated string (e.g. 2 or 2-2-2).",
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
        Optional[Vec3Int],
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
        Optional[str],
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string \
(e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Converts a RAW file into a WEBKNOSSOS dataset."""

    if flip_axes is not None:
        for index in flip_axes:
            assert (
                0 <= index <= 3
            ), "flip_axes parameter must only contain indices between 0 and 3."

    mode = SamplingModes.parse(sampling_mode.value)
    if source.is_dir():
        logger.error("source_path is not a file")
        return

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    mag_view = convert_raw(
        source,
        target,
        layer_name,
        dtype,
        shape,
        data_format,
        chunk_shape,
        chunks_per_shard,
        order.value,
        voxel_size,
        flip_axes,
        compress,
        executor_args,
    )

    with get_executor_for_args(executor_args) as executor:
        mag_view.layer.downsample(
            from_mag=mag_view.mag,
            coarsest_mag=max_mag,
            interpolation_mode=interpolation_mode,
            compress=compress,
            sampling_mode=mode,
            executor=executor,
        )

"""This module converts an image stack to a WEBKNOSSOS dataset."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, Optional

import typer
from typing_extensions import Annotated

from webknossos.dataset.length_unit import LengthUnit
from webknossos.dataset.properties import DEFAULT_LENGTH_UNIT_STR, VoxelSize
from webknossos.geometry.vec3_int import Vec3Int

from ..dataset import DataFormat, Dataset
from ..utils import get_executor_for_args
from ._utils import (
    DistributionStrategy,
    LayerCategory,
    VoxelSizeTuple,
    parse_path,
    parse_voxel_size,
)


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your image data.",
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
    layer_name: Annotated[
        Optional[str],
        typer.Option(
            help="This name is used if only one layer is created. Otherwise this name is used as a common prefix for all layers.",
        ),
    ] = None,
    category: Annotated[
        Optional[LayerCategory],
        typer.Option(
            help="The category of the layer that should be created.",
        ),
    ] = None,
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = "wkw",  # type:ignore
    name: Annotated[
        Optional[str],
        typer.Option(
            help="New name for the WEBKNOSSOS dataset "
            "(if not provided, final component of target path is used)"
        ),
    ] = None,
    chunk_shape: Annotated[
        Optional[str],
        typer.Option(
            help="Chunk shape of the target dataset. Should be a comma separated string "
            "(e.g. 64,64,64).",
            parser=Vec3Int.from_str,
            metavar="Vec3Int",
        ),
    ] = None,
    chunks_per_shard: Annotated[
        Optional[str],
        typer.Option(
            help="Shard shape of the target dataset. Should be a comma separated string "
            "(e.g. 64,64,64).",
            parser=Vec3Int.from_str,
            metavar="Vec3Int",
        ),
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
    ] = False,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            help="Number of images to be processed in one batch (influences RAM consumption). "
            "When creating a WKW dataset, batch-size must be a multiple of chunk-shape's z dimension. "
            "When converting to Zarr, batch-size must be a multiple of the z dimension of the "
            "shard shape (chunk-shape x chunks-per-shard).",
        ),
    ] = None,
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
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Automatic detection of an image stack and conversion to a WEBKNOSSOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    voxel_size_with_unit = VoxelSize(voxel_size, unit)

    with get_executor_for_args(args=executor_args) as executor:
        Dataset.from_images(
            source,
            target,
            name=name,
            voxel_size_with_unit=voxel_size_with_unit,
            chunk_shape=chunk_shape,  # type: ignore
            chunks_per_shard=chunks_per_shard,  # type: ignore
            data_format=data_format,
            executor=executor,
            compress=compress,
            layer_name=layer_name,
            batch_size=batch_size,
            layer_category=category.value if category else None,
        )

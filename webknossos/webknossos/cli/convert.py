"""This module converts an image stack to a WEBKNOSSOS dataset."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Any, Optional

import typer
from typing_extensions import Annotated

from ..dataset import DataFormat, Dataset
from ..utils import get_executor_for_args
from ._utils import (
    DistributionStrategy,
    LayerCategory,
    VoxelSize,
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
        VoxelSize,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma seperated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VOXEL_SIZE",
        ),
    ],
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
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
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
        Optional[str],
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job_resources=\'{"mem": "10M"}\')\'',
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

    with get_executor_for_args(args=executor_args) as executor:
        Dataset.from_images(
            source,
            target,
            voxel_size,
            name,
            data_format=data_format,
            executor=executor,
            compress=compress,
            layer_category=category.value if category else None,
        )

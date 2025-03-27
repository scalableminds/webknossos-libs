"""This module copies a WEBKNOSSOS datasets."""

import logging
from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated, Any

import typer

from ..dataset import DataFormat, Dataset
from ..geometry import Vec3Int
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, parse_path, parse_vec3int

logger = logging.getLogger(__name__)


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to the source WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    target: Annotated[
        Any,
        typer.Argument(
            help="Path to the target WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    data_format: Annotated[
        DataFormat | None,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = None,
    chunk_shape: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the target dataset "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    shard_shape: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Number of voxels to be stored as a shard in the target dataset "
            "(e.g. `1024` or `1024,1024,1024`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    exists_ok: Annotated[
        bool, typer.Option(help="Whether it should overwrite an existing dataset.")
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
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Make a copy of the WEBKNOSSOS dataset.

    Remote paths (i.e. https and s3) are also allowed.
    Use the following environment variables to configure remote paths:
    - HTTP_BASIC_USER
    - HTTP_BASIC_PASSWORD
    - S3_ENDPOINT_URL
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    """

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    source_dataset = Dataset.open(source)

    with get_executor_for_args(args=executor_args) as executor:
        source_dataset.copy_dataset(
            target,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            data_format=data_format,
            exists_ok=exists_ok,
            executor=executor,
        )

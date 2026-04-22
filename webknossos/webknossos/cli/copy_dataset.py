"""This module copies a WEBKNOSSOS datasets."""

import logging
from typing import Annotated, Any

import typer

from ..dataset import Dataset
from ..dataset_properties import DataFormat
from ..geometry import Vec3Int
from ._utils import (
    DistributionStrategy,
    DistributionStrategyOption,
    ExistsOkOption,
    JobResourcesOption,
    JobsOption,
    ShardShapeOption,
    get_executor_for_args,
    parse_path,
    parse_vec3int,
)

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
    shard_shape: ShardShapeOption = None,
    exists_ok: ExistsOkOption = False,
    jobs: JobsOption = None,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
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

    source_dataset = Dataset.open(source)

    with get_executor_for_args(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    ) as executor:
        source_dataset.copy_dataset(
            target,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            data_format=data_format,
            exists_ok=exists_ok,
            executor=executor,
        )

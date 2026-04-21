"""This module merges a volume annotation layer with its fallback layer."""

from typing import Annotated, Any

import typer

from ..annotation import Annotation
from ._utils import (
    DEFAULT_JOBS,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    get_executor_for_args,
    parse_path,
)


def main(
    *,
    target: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS output dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    source_annotation: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS zip annotation",
            show_default=False,
            parser=parse_path,
        ),
    ],
    dataset_directory: Annotated[
        Any,
        typer.Argument(
            help="Path to your WEBKNOSSOS dataset folder.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    volume_layer_name: Annotated[
        str | None,
        typer.Option(help="Name of the volume layer to merge with fallback layer."),
    ] = None,
    jobs: JobsOption = DEFAULT_JOBS,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
) -> None:
    """Merges a given WEBKNOSSOS annotation with a volume annotation of a fallback layer."""

    with get_executor_for_args(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    ) as executor:
        Annotation.load(source_annotation).merge_fallback_layer(
            target=target,
            dataset_directory=dataset_directory,
            volume_layer_name=volume_layer_name,
            executor=executor,
        )

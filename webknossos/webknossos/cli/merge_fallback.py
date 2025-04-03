"""This module merges a volume annotation layer with its fallback layer."""

from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated, Any

import typer

from ..annotation import Annotation
from ..utils import get_executor_for_args
from ._utils import DistributionStrategy, parse_path


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
    """Merges a given WEBKNOSSOS annotation with a volume annotation of a fallback layer."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )

    with get_executor_for_args(args=executor_args) as executor:
        Annotation.load(source_annotation).merge_fallback_layer(
            target=target,
            dataset_directory=dataset_directory,
            volume_layer_name=volume_layer_name,
            executor=executor,
        )

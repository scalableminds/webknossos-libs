"""This module takes care of compressing WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

from webknossos import Dataset
from webknossos.utils import get_executor_for_args
from wkcuber.utils import DistributionStrategy


def main(
    *,
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ],
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
    """Compress a given WEBKNOSSOS dataset."""

    rprint(f"Compressing [blue]{target}[/blue] ...")

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    )

    with get_executor_for_args(args=executor_args) as executor:
        Dataset.open(target).compress(executor=executor)

    rprint("[bold green]Done.[/bold green]")

"""This module takes care of upsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Optional, Tuple

import typer
from rich import print as rprint

from webknossos import Dataset, Mag, Vec3Int
from webknossos.utils import get_executor_for_args
from wkcuber.utils import DistributionStrategy

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ],
    from_mag: Annotated[
        Tuple[int, int, int],
        typer.Option(help="Mag to start upsampling"),
    ],
    layer_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of the layer that should be downsampled.", show_default=False
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
    job_ressources: Annotated[
        Optional[str],
        typer.Option(
            help='Necessary when using slurm as distribution strategy. Should be a JSON string (e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Upsample your WEBKNOSSOS dataset."""

    rprint(f"Doing upsampling for [blue]{target}[/blue] ...")

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_ressources=job_ressources,
    )
    dataset = Dataset.open(target)
    mag = Mag(Vec3Int.from_xyz(*from_mag))
    with get_executor_for_args(args=executor_args) as executor:
        if layer_name is None:
            upsample_all_layers(dataset, mag, executor_args)
        else:
            layer = dataset.get_layer(layer_name)
            layer.upsample(from_mag=mag, executor=executor)

    rprint("[bold green]Done.[/bold green]")


def upsample_all_layers(
    dataset: Dataset, from_mag: Mag, executor_args: Namespace
) -> None:
    """Iterates over all layers and upsamples them."""

    for layer in dataset.layers.values():
        with get_executor_for_args(args=executor_args) as executor:
            layer.upsample(from_mag=from_mag, executor=executor)

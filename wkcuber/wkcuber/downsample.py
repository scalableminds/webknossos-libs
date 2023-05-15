"""This module takes care of downsampling WEBKNOSSOS datasets."""

from argparse import Namespace
from multiprocessing import cpu_count
from pathlib import Path
from typing import Annotated, Optional, Union

import typer
from rich import print as rprint

from webknossos import Dataset
from webknossos.utils import get_executor_for_args
from wkcuber.utils import DistributionStrategy

app = typer.Typer(
    invoke_without_command=True,
)


@app.callback()
def main(
    *,
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ],
    # sampling_mode: Annotated[
    #     SamplingModes, typer.Option(help="The sampling mode to use.")
    # ] = SamplingModes.ANISOTROPIC,
    layer_name: Annotated[
        Union[str, None],
        typer.Option(help="Name of the layer that should be downsampled."),
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
            help='Necessary when using slurm as distribution strategy. Should be a JSON string \
                (e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Downsample your WEBKNOSSOS dataset."""

    rprint(f"Doing downsampling for [blue]{target}[/blue] ...")

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_ressources=job_ressources,
    )

    dataset = Dataset.open(target)
    with get_executor_for_args(args=executor_args) as executor:
        if layer_name is None:
            dataset.downsample(executor=executor)
        else:
            layer = dataset.get_layer(layer_name)
            layer.downsample(executor=executor)

    rprint("[bold green]Done.[/bold green]")

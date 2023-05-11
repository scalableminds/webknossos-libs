"""This module takes care of downsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, Union

import typer
from rich import print as rprint
from webknossos import Dataset

app = typer.Typer(
    invoke_without_command=True,
)


@app.callback()
def main(
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ],
    layer_name: Annotated[
        Union[str, None],
        typer.Argument(help="Name of the layer that should be downsampled."),
    ] = None,
) -> None:
    """Downsample your WEBKNOSSOS dataset."""

    rprint(f"Doing downsampling for [blue]{target}[/blue] ...")

    try:
        dataset = Dataset.open(target)
        if layer_name is None:
            dataset.downsample()
        else:
            layer = dataset.get_layer(layer_name)
            layer.downsample()

        rprint("[bold green]Done.[/bold green]")

    except AssertionError as err:
        rprint(f"[bold red]Error![/bold red] {err}")

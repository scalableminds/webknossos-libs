"""This module takes care of upsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, Optional, Tuple

import typer
from rich import print as rprint
from webknossos import Dataset, Mag, Vec3Int

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
) -> None:
    """Upsample your WEBKNOSSOS dataset."""

    rprint(f"Doing upsampling for [blue]{target}[/blue] ...")

    dataset = Dataset.open(target)
    mag = Mag(Vec3Int.from_xyz(*from_mag))
    if layer_name is None:
        upsample_all_layers(dataset, mag)
    else:
        layer = dataset.get_layer(layer_name)
        layer.upsample(from_mag=mag)

    rprint("[bold green]Done.[/bold green]")


def upsample_all_layers(dataset: Dataset, from_mag: Mag) -> None:
    for layer in dataset.layers.values():
        layer.upsample(from_mag=from_mag)

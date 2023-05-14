"""This module takes care of downsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import typer
from rich import print as rprint

from webknossos import BoundingBox, Dataset

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    path: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Path to save your WEBKNOSSOS dataset.",
        ),
    ],
    exist_ok: Annotated[
        bool,
        typer.Option(
            help="Is it alright to overwrite existing data on download path.",
        ),
    ] = False,
    full_url: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Args for full URL download",
            help="URL where your dataset is available.",
        ),
    ] = None,
    dataset_name: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Download with dataset name",
            help="Name of your dataset on your WEBKNOSSOS instance.",
        ),
    ] = None,
    organisation_id: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Download with dataset name",
            help="Your organization id.",
        ),
    ] = None,
    sharing_token: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Download with dataset name",
            help="A sharing token for the dataset.",
        ),
    ] = None,
    webknossos_url: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Download with dataset name",
            help="URL where your WEBKNOSSOS instance is hosted.",
        ),
    ] = "https://webknossos.org",
    bounding_box: Annotated[
        Optional[Tuple[int, int, int, int, int, int]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Bounding box that should be downloaded. First three integers are top-left coordinates and the remaining integers are bottom-right coordinates.",
        ),
    ] = None,
    layers: Annotated[
        Optional[List[str]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Layers that should be downloaded.",
        ),
    ] = None,
) -> None:
    """Download a WEBKNOSSOS dataset from a remote location."""

    bbox = None if bounding_box is None else BoundingBox.from_tuple6(bounding_box)

    if full_url is not None:
        rprint(f"Downloading from url [blue]{full_url}[/blue] ...")
        Dataset.download(
            dataset_name_or_url=full_url,
            bbox=bbox,
            layers=layers,
            path=path,
            exist_ok=exist_ok,
        )
    elif dataset_name is not None:
        rprint(
            f"Downloading [blue]{dataset_name}[/blue] from [blue]{webknossos_url}[/blue] ..."
        )
        Dataset.download(
            dataset_name_or_url=dataset_name,
            organization_id=organisation_id,
            sharing_token=sharing_token,
            webknossos_url=webknossos_url,
            bbox=bbox,
            layers=layers,
            path=path,
            exist_ok=exist_ok,
        )

    rprint("[bold green]Done.[/bold green]")

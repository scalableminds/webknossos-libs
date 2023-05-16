"""This module takes care of downsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, List, Optional

import typer
from rich import print as rprint

from webknossos import BoundingBox, Dataset, Mag, webknossos_context
from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from wkcuber.utils import parse_bbox, parse_mag

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    *,
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
            envvar="WK_URL",
        ),
    ] = DEFAULT_WEBKNOSSOS_URL,
    token: Annotated[
        Optional[str],
        typer.Option(
            help="Authentication token for WEBKNOSSOS instance.",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    bbox: Annotated[
        Optional[BoundingBox],
        typer.Option(
            rich_help_panel="Partial download",
            help="Bounding box that should be downloaded. First three integers are top-left \
            coordinates and the remaining integers are bottom-right coordinates.",
            parser=parse_bbox,
        ),
    ] = None,
    layer: Annotated[
        Optional[List[str]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Layers that should be downloaded.",
        ),
    ] = None,
    mag: Annotated[
        Optional[List[Mag]],
        typer.Option(
            help="Mags that should be downloaded, e.g. Mag 1 and 2 with --mag 1 --mag 2.",
            parser=parse_mag,
        ),
    ] = None,
) -> None:
    """Download a WEBKNOSSOS dataset from a remote location."""

    if full_url is not None:
        rprint(f"Downloading from url [blue]{full_url}[/blue] ...")
        with webknossos_context(url=webknossos_url, token=token):
            Dataset.download(
                dataset_name_or_url=full_url,
                bbox=bbox,
                layers=layer,
                path=path,
                exist_ok=exist_ok,
                mags=mag,
            )
    elif dataset_name is not None:
        rprint(
            f"Downloading [blue]{dataset_name}[/blue] from [blue]{webknossos_url}[/blue] ..."
        )
        with webknossos_context(url=webknossos_url, token=token):
            Dataset.download(
                dataset_name_or_url=dataset_name,
                organization_id=organisation_id,
                sharing_token=sharing_token,
                webknossos_url=webknossos_url,
                bbox=bbox,
                layers=layer,
                path=path,
                exist_ok=exist_ok,
                mags=mag,
            )

    rprint("[bold green]Done.[/bold green]")

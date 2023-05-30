"""This module takes care of downloading WEBKNOSSOS datasets."""

from typing import Any, List, Optional

import typer
from rich import print as rprint
from typing_extensions import Annotated

from webknossos import BoundingBox, Dataset, Mag, webknossos_context
from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from wkcuber._utils import parse_bbox, parse_mag, parse_path


def main(
    *,
    target: Annotated[
        Any,
        typer.Argument(
            show_default=False,
            help="Path to save your WEBKNOSSOS dataset.",
            parser=parse_path,
            metavar="PATH",
        ),
    ],
    exist_ok: Annotated[
        bool,
        typer.Option(
            help="Allow overwriting of dataset if it already exists on disk.",
        ),
    ] = False,
    full_url: Annotated[
        Optional[str],
        typer.Option(
            rich_help_panel="Args for full URL download",
            help="WEBKNOSSOS URL of your dataset.",
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
            help="Authentication token for WEBKNOSSOS instance \
(https://webknossos.org/auth/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    bbox: Annotated[
        Optional[BoundingBox],
        typer.Option(
            rich_help_panel="Partial download",
            help="Bounding box that should be downloaded. \
Should be a comma seperated string (e.g. 0,0,0,10,10,10)",
            parser=parse_bbox,
            metavar="BBOX",
        ),
    ] = None,
    layer: Annotated[
        Optional[List[str]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Layers that should be downloaded. \
For multiple layers type: --layer color --layer segmentation",
        ),
    ] = None,
    mag: Annotated[
        Optional[List[Mag]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Mags that should be downloaded. \
Should be number or minus seperated string (e.g. 2 or 2-2-2). \
For multiple mags type: --mag 1 --mag 2",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = None,
) -> None:
    """Download a dataset from a WEBKNOSSOS server."""

    layers = layer if layer else None
    mags = mag if mag else None

    if full_url is not None:
        with webknossos_context(url=webknossos_url, token=token):
            Dataset.download(
                dataset_name_or_url=full_url,
                path=target,
                exist_ok=exist_ok,
                bbox=bbox,
                layers=layers,
                mags=mags,
            )
    elif dataset_name is not None:
        with webknossos_context(url=webknossos_url, token=token):
            Dataset.download(
                dataset_name_or_url=dataset_name,
                organization_id=organisation_id,
                sharing_token=sharing_token,
                webknossos_url=webknossos_url,
                bbox=bbox,
                layers=layers,
                path=target,
                exist_ok=exist_ok,
                mags=mags,
            )
    else:
        raise ValueError(
            "Either define a full-url for downloading or specify your dataset with a name."
        )

    rprint("[bold green]Done.[/bold green]")

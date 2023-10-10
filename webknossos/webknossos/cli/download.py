"""This module takes care of downloading WEBKNOSSOS datasets."""

from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from ..annotation import Annotation
from ..client import webknossos_context
from ..dataset import Dataset
from ..geometry import BoundingBox, Mag
from ._utils import parse_bbox, parse_mag, parse_path


def main(
    *,
    target: Annotated[
        Any,
        typer.Argument(
            show_default=False,
            help="Path to save your WEBKNOSSOS dataset.",
            parser=parse_path,
        ),
    ],
    url: Annotated[
        str,
        typer.Option(
            help="URL of your dataset or your annotation.",
        ),
    ],
    token: Annotated[
        Optional[str],
        typer.Option(
            help="Authentication token for WEBKNOSSOS instance "
            "(https://webknossos.org/auth/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    bbox: Annotated[
        Optional[BoundingBox],
        typer.Option(
            rich_help_panel="Partial download",
            help="Bounding box that should be downloaded. "
            "The input format is x,y,z,width,height,depth. "
            "Should be a comma seperated string (e.g. 0,0,0,10,10,10).",
            parser=parse_bbox,
            metavar="BBOX",
        ),
    ] = None,
    layer: Annotated[
        Optional[List[str]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Layers that should be downloaded. "
            "For multiple layers type: --layer color --layer segmentation",
        ),
    ] = None,
    mag: Annotated[
        Optional[List[Mag]],
        typer.Option(
            rich_help_panel="Partial download",
            help="Mags that should be downloaded. "
            "Should be number or minus seperated string (e.g. 2 or 2-2-2). "
            "For multiple mags type: --mag 1 --mag 2",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = None,
) -> None:
    """Download a dataset from a WEBKNOSSOS server."""

    layers = layer if layer else None
    mags = mag if mag else None

    with webknossos_context(token=token):
        try:
            Dataset.download(
                dataset_name_or_url=url,
                path=target,
                bbox=bbox,
                layers=layers,
                mags=mags,
            )
        except AssertionError:
            Annotation.download(annotation_id_or_url=url).save(target)

"""This module takes care of downloading WEBKNOSSOS datasets."""

import re
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from ..annotation.annotation import _ANNOTATION_URL_REGEX, Annotation
from ..client import webknossos_context
from ..client._resolve_short_link import resolve_short_link
from ..dataset.dataset import _DATASET_DEPRECATED_URL_REGEX, _DATASET_URL_REGEX, Dataset
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
            "Should be a comma separated string (e.g. 0,0,0,10,10,10).",
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
            "Should be number or minus separated string (e.g. 2 or 2-2-2). "
            "For multiple mags type: --mag 1 --mag 2",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = None,
) -> None:
    """Download a dataset from a WEBKNOSSOS server."""

    layers = layer if layer else None
    mags = mag if mag else None
    url = resolve_short_link(url)

    with webknossos_context(token=token):
        if re.match(_DATASET_URL_REGEX, url) or re.match(
            _DATASET_DEPRECATED_URL_REGEX, url
        ):
            Dataset.download(
                dataset_name_or_url=url,
                path=target,
                bbox=bbox,
                layers=layers,
                mags=mags,
            )
        elif re.match(_ANNOTATION_URL_REGEX, url):
            Annotation.download(annotation_id_or_url=url).save(target)
        else:
            raise RuntimeError(
                "The provided URL does not lead to a dataset or annotation."
            )


if __name__ == "__main__":
    typer.run(main)

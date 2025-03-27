"""This module takes care of uploading datasets to a WEBKNOSSOS server."""

from typing import Annotated, Any

import typer

from ..client import webknossos_context
from ..client._defaults import DEFAULT_WEBKNOSSOS_URL
from ..client._upload_dataset import DEFAULT_SIMULTANEOUS_UPLOADS
from ..dataset import Dataset
from ._utils import parse_path


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            show_default=False,
            help="Path to your local WEBKNOSSOS dataset.",
            parser=parse_path,
        ),
    ],
    webknossos_url: Annotated[
        str,
        typer.Option(
            help="URL to WEBKNOSSOS instance.",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_URL",
        ),
    ] = DEFAULT_WEBKNOSSOS_URL,
    token: Annotated[
        str | None,
        typer.Option(
            help="Authentication token for WEBKNOSSOS instance "
            "(https://webknossos.org/auth/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option(
            help="Alternative name to rename your dataset on upload to WEBKNOSSOS. "
            "(if not provided, current name of dataset is used)",
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = DEFAULT_SIMULTANEOUS_UPLOADS,
) -> None:
    """Upload a dataset to a WEBKNOSSOS server."""

    with webknossos_context(url=webknossos_url, token=token):
        Dataset.open(dataset_path=source).upload(
            new_dataset_name=dataset_name, jobs=jobs
        )

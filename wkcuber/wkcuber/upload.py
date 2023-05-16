"""This module takes care of downsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

from webknossos import Dataset, webknossos_context
from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client._upload_dataset import DEFAULT_SIMULTANEOUS_UPLOADS

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    *,
    path: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Path to your local WEBKNOSSOS dataset.",
        ),
    ],
    url: Annotated[
        str,
        typer.Option(
            help="URL to WEBKNOSSOS instance.",
            rich_help_panel="WEBKNOSSOS context",
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
    dataset_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of your dataset on your WEBKNOSSOS.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of simultaneous chunk uploads.",
            rich_help_panel="Executor options",
        ),
    ] = DEFAULT_SIMULTANEOUS_UPLOADS,
) -> None:
    """Upload the WEBKNOSSOS dataset to a remote location."""

    rprint(f"Uploading [blue]{dataset_name}[/blue] ...")

    with webknossos_context(url=url, token=token):
        Dataset.open(dataset_path=path).upload(new_dataset_name=dataset_name, jobs=jobs)

    rprint("[bold green]Done.[/bold green]")

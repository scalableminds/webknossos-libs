"""This module takes care of downsampling WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint

from webknossos import Dataset

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    path: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Path to your local WEBKNOSSOS dataset.",
        ),
    ],
    dataset_name: Annotated[
        Optional[str],
        typer.Option(
            help="Name of your dataset on your WEBKNOSSOS.",
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of simultaneous chunk uploads.",
            rich_help_panel="Executor options",
        ),
    ] = 5,
):
    """Upload the WEBKNOSSOS dataset to a remote location."""

    rprint(f"Uploading [blue]{dataset_name}[/blue] ...")

    dataset = Dataset.open(dataset_path=path)
    dataset.upload(new_dataset_name=dataset_name, jobs=jobs)

    rprint("[bold green]Done.[/bold green]")

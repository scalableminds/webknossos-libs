"""This module checks equality of two different WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from webknossos import Dataset

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to your first WEBKNOSSOS dataset.", show_default=False
        ),
    ],
    target: Annotated[
        Path,
        typer.Argument(
            help="Path to your second WEBKNOSSOS dataset.", show_default=False
        ),
    ],
):
    """[Not supported yet] Check equality of two WEBKNOSSOS datasets."""

    rprint(f"Comparing [blue]{source}[/blue] with [blue]{target}[/blue] ...")

    if Dataset.open(source) == Dataset.open(target):
        rprint("[green]The two datasets are equal.[/green]")
    else:
        rprint("[red]Not equal[/red]")

    rprint("[bold green]Done.[/bold green]")

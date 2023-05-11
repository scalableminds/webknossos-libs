"""This module takes care of compressing WEBKNOSSOS datasets."""

from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from webknossos import Dataset

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main(
    target: Annotated[
        Path,
        typer.Argument(help="Path to your WEBKNOSSOS dataset.", show_default=False),
    ]
) -> None:
    """Compress a given WEBKNOSSOS dataset."""

    rprint(f"Compressing [blue]{target}[/blue] ...")

    Dataset.open(target).compress()

    rprint("[bold green]Done.[/bold green]")

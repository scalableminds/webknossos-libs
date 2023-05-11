"""This module offers functionality to export your WEBKNOSSOS dataset."""

import typer
from rich import print as rprint

app = typer.Typer(invoke_without_command=True)


@app.callback()
def main() -> None:
    """Export your WEBKNOSSOS datasets to the output format of your needs."""
    print("I export stuff.")
    rprint("[bold green]Done.[/bold green]")

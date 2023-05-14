"""This module delivers a CLI to work with WEBKNOSSOS datasets."""
import typer

from wkcuber import (
    check_equality,
    compress,
    convert,
    download,
    downsample,
    upload,
    upsample,
)

app = typer.Typer(no_args_is_help=True)

app.add_typer(convert.app, name="convert", subcommand_metavar="")
app.add_typer(downsample.app, name="downsample", subcommand_metavar="")
app.add_typer(upsample.app, name="upsample", subcommand_metavar="")
app.add_typer(compress.app, name="compress", subcommand_metavar="")
app.add_typer(check_equality.app, name="check-equality", subcommand_metavar="")
app.add_typer(download.app, name="download", subcommand_metavar="")
app.add_typer(upload.app, name="upload", subcommand_metavar="")

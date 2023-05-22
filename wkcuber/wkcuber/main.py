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

app.command("check_equality")(check_equality.main)
app.command("compress")(compress.main)
app.command("convert")(convert.main)
app.command("download")(download.main)
app.command("downsample")(downsample.main)
app.command("upload")(upload.main)
app.command("upsample")(upsample.main)

"""This module delivers a CLI to work with WEBKNOSSOS datasets."""

import typer

from . import (
    check_equality,
    compress,
    convert,
    convert_knossos,
    convert_raw,
    convert_zarr,
    copy_dataset,
    download,
    downsample,
    export_as_tiff,
    merge_fallback,
    upload,
    upsample,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_short=False)

app.command("check-equality")(check_equality.main)
app.command("compress")(compress.main)
app.command("convert")(convert.main)
app.command("convert-knossos")(convert_knossos.main)
app.command("convert-raw")(convert_raw.main)
app.command("convert-zarr")(convert_zarr.main)
app.command("copy-dataset")(copy_dataset.main)
app.command("download")(download.main)
app.command("downsample")(downsample.main)
app.command("export-wkw-as-tiff")(export_as_tiff.main)
app.command("export-as-tiff")(export_as_tiff.main)
app.command("merge-fallback")(merge_fallback.main)
app.command("upload")(upload.main)
app.command("upsample")(upsample.main)

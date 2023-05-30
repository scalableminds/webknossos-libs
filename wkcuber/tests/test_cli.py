"""This module containes tests for the wkcuber CLI."""

import json
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Iterator, Union

import pytest
from typer.testing import CliRunner
from upath import UPath

from webknossos import DataFormat, Dataset
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from wkcuber.main import app

runner = CliRunner()


@contextmanager
def tmp_cwd() -> Iterator[None]:
    """Creates a temporary working directory to test side effects."""

    prev_cwd = os.getcwd()
    with TemporaryDirectory() as new_cwd:
        os.chdir(new_cwd)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


MINIO_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
MINIO_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
MINIO_PORT = "8000"

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


@pytest.fixture(scope="module")
def remote_testoutput_path() -> UPath:
    """Minio is an S3 clone and is used as local test server"""
    container_name = "minio"
    cmd = (
        "docker run"
        f" -p {MINIO_PORT}:9000"
        f" -e MINIO_ROOT_USER={MINIO_ROOT_USER}"
        f" -e MINIO_ROOT_PASSWORD={MINIO_ROOT_PASSWORD}"
        f" --name {container_name}"
        " --rm"
        " -d"
        " minio/minio server /data"
    )
    subprocess.check_output(shlex.split(cmd))
    remote_path = UPath(
        "s3://testoutput",
        key=MINIO_ROOT_USER,
        secret=MINIO_ROOT_PASSWORD,
        client_kwargs={"endpoint_url": f"http://localhost:{MINIO_PORT}"},
    )
    remote_path.fs.mkdirs("testoutput", exist_ok=True)
    try:
        yield remote_path
    finally:
        subprocess.check_output(["docker", "stop", container_name])


def check_call(*args: Union[str, int, Path]) -> None:
    try:
        subprocess.check_call([str(a) for a in args])
    except subprocess.CalledProcessError as err:
        print(f"Process failed with exit code {err.returncode}: `{args}`")
        raise err


def _tiff_cubing(out_path: Path, data_format: DataFormat) -> None:
    in_path = TESTDATA_DIR / "tiff"

    check_call(
        "wkcuber",
        "convert",
        "--jobs",
        2,
        "--voxel-size",
        "11.24,11.24,25",
        "--data-format",
        str(data_format),
        in_path,
        out_path,
    )

    assert (out_path / "tiff").exists()
    assert (out_path / "tiff" / "1").exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Only run this test on Linux, because it requires a running `minio` docker container.",
)
def test_tiff_cubing_zarr_s3(remote_testoutput_path: UPath) -> None:
    out_path = remote_testoutput_path / "tiff_cubing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_ROOT_PASSWORD
    os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ROOT_USER
    os.environ["S3_ENDPOINT_URL"] = f"http://localhost:{MINIO_PORT}"

    _tiff_cubing(out_path, DataFormat.Zarr)

    assert (out_path / "tiff" / "1" / ".zarray").exists()
    assert (out_path / PROPERTIES_FILE_NAME).exists()

    with (out_path / PROPERTIES_FILE_NAME).open("r") as a, (
        TESTDATA_DIR / "tiff" / "datasource-properties.zarr-fixture.json"
    ).open("r") as fixture:
        json_a = json.load(a)
        json_fixture = json.load(fixture)
        del json_a["id"]
        del json_fixture["id"]
        assert json_a == json_fixture


def test_main() -> None:
    """Tests the functionality of the wkcuber command."""

    result_without_args = runner.invoke(app, [])
    assert result_without_args.exit_code == 0


def test_check_equality() -> None:
    """Tests the functionality of check_equality subcommand."""

    result_without_args = runner.invoke(app, ["check-equality"])
    assert result_without_args.exit_code == 2

    result = runner.invoke(
        app,
        [
            "check-equality",
            "testdata/simple_wkw_dataset",
            "testdata/simple_wkw_dataset",
        ],
    )
    assert result.exit_code == 0
    assert (
        "The datasets testdata/simple_wkw_dataset and testdata/simple_wkw_dataset are equal"
        in result.stdout.replace("\n", "")
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compress() -> None:
    """Tests the functionality of compress subcommand."""

    result_without_args = runner.invoke(app, ["compress"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = TESTDATA_DIR / "simple_wkw_dataset"
        copytree(wkw_path, Path("testdata") / "simple_wkw_dataset")

        result = runner.invoke(app, ["compress", "testdata/simple_wkw_dataset"])

        assert result.exit_code == 0
        assert "Done." in result.stdout


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_convert() -> None:
    """Tests the functionality of convert subcommand."""

    result_without_args = runner.invoke(app, ["convert"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        origin_path = TESTDATA_DIR / "tiff"
        wkw_path = Path("new_wkw_dataset")

        result = runner.invoke(
            app,
            [
                "convert",
                "--voxel-size",
                "11.0,11.0,11.0",
                str(origin_path),
                str(wkw_path),
            ],
        )

        assert result.exit_code == 0
        assert (wkw_path / "datasource-properties.json").exists()


def test_download() -> None:
    """Tests the functionality of download subcommand."""

    result = runner.invoke(app, ["download"])
    assert result.exit_code == 2

    with tmp_cwd():
        result = runner.invoke(
            app,
            [
                "download",
                "--bbox",
                "0,0,0,5,5,5",
                "--mag",
                "8",
                "--full-url",
                "https://webknossos.org/datasets/scalable_minds/cremi_example/",
                "testoutput/",
            ],
        )
        assert result.exit_code == 0
        assert (Path("testoutput") / "datasource-properties.json").exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_downsample_and_upsample() -> None:
    """Tests the functionality of downsample subcommand."""

    result_without_args = runner.invoke(app, ["downsample"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = Path("simple_wkw_dataset")
        copytree(TESTDATA_DIR / wkw_path, wkw_path)

        result_downsample = runner.invoke(app, ["downsample", str(wkw_path)])

        assert result_downsample.exit_code == 0
        assert (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()
        assert (wkw_path / "color" / "2" / "z0" / "y0" / "x0.wkw").exists()
        assert (wkw_path / "color" / "4" / "z0" / "y0" / "x0.wkw").exists()

        Dataset.open(wkw_path).get_layer("color").delete_mag(1)

        assert not (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()

        result_upsample = runner.invoke(
            app, ["upsample", "--from-mag", "2", str(wkw_path)]
        )

        assert result_upsample.exit_code == 0
        assert (wkw_path / "color" / "1" / "z0" / "y0" / "x0.wkw").exists()


def test_upload() -> None:
    """Tests the functionality of upload subcommand."""

    result_without_args = runner.invoke(app, ["upload"])
    assert result_without_args.exit_code == 2


def test_upsample() -> None:
    """Tests the functionality of upsample subcommand."""

    result = runner.invoke(app, ["upsample"])
    assert result.exit_code == 2

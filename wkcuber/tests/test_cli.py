"""This module containes tests for the wkcuber CLI."""

import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest
from typer.testing import CliRunner

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
    assert "The two datasets are equal." in result.stdout


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compress() -> None:
    """Tests the functionality of compress subcommand."""

    result_without_args = runner.invoke(app, ["compress"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = Path(__file__).parent.parent / "testdata" / "simple_wkw_dataset"
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
        origin_path = Path(__file__).parent.parent / "testdata" / "tiff"
        wkw_path = Path("new_wkw_dataset")

        result = runner.invoke(
            app,
            [
                "convert",
                "--voxel-size",
                "11",
                "11",
                "11",
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
                "--bounding-box",
                "0",
                "0",
                "0",
                "5",
                "5",
                "5",
                "--mag",
                "8",
                "8",
                "8",
                "--full-url",
                "https://webknossos.org/datasets/scalable_minds/cremi_example/",
                "testoutput/",
            ],
        )
        assert result.exit_code == 0
        assert (Path("testoutput") / "datasource-properties.json").exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_downsample() -> None:
    """Tests the functionality of downsample subcommand."""

    result_without_args = runner.invoke(app, ["downsample"])
    assert result_without_args.exit_code == 2

    with tmp_cwd():
        wkw_path = Path("testdata") / "simple_wkw_dataset"
        copytree(Path(__file__).parent.parent / wkw_path, wkw_path)

        result = runner.invoke(app, ["downsample", str(wkw_path)])

        assert result.exit_code == 0
        assert (wkw_path / "color" / "2" / "z0" / "y0" / "x0.wkw").exists()
        assert (wkw_path / "color" / "4" / "z0" / "y0" / "x0.wkw").exists()


def test_upload() -> None:
    """Tests the functionality of upload subcommand."""

    result_without_args = runner.invoke(app, ["upload"])
    assert result_without_args.exit_code == 2


def test_upsample() -> None:
    """Tests the functionality of upsample subcommand."""

    result = runner.invoke(app, ["upsample"])
    assert result.exit_code == 2

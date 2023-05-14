from typer.testing import CliRunner

from wkcuber.main import app

runner = CliRunner()


def test_check_equality() -> None:
    """Tests the functionality of check_equality subcommand."""
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


def test_compress() -> None:
    """Tests the functionality of compress subcommand."""

    result = runner.invoke(app, ["compress"])
    assert result.exit_code == 2


def test_convert() -> None:
    """Tests the functionality of convert subcommand."""

    result = runner.invoke(app, ["convert"])
    assert result.exit_code == 2


def test_download() -> None:
    """Tests the functionality of download subcommand."""

    result = runner.invoke(app, ["download"])
    assert result.exit_code == 2


def test_downsample() -> None:
    """Tests the functionality of downsample subcommand."""

    result = runner.invoke(app, ["downsample"])
    assert result.exit_code == 2


def test_upload() -> None:
    """Tests the functionality of upload subcommand."""

    result = runner.invoke(app, ["upload"])
    assert result.exit_code == 2


def test_upsample() -> None:
    """Tests the functionality of upsample subcommand."""

    result = runner.invoke(app, ["upsample"])
    assert result.exit_code == 2

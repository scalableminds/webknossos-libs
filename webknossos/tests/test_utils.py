from shutil import ignore_patterns
from unittest.mock import Mock, patch

import pytest
from upath import UPath

from tests.utils import TestTemporaryDirectoryNonLocal
from webknossos.utils import call_with_retries, copytree, dump_path


def test_call_with_retries_success() -> None:
    """Test that a successful function call returns immediately."""
    mock_fn = Mock(return_value="success")

    result = call_with_retries(mock_fn)

    assert result == "success"
    mock_fn.call_count == 1  # Function called only once (direct success)


@patch("time.sleep")
def test_call_with_retries_sucess_with_retry(mock_sleep: Mock) -> None:
    """Test retry behavior when function succeeds after retryable failures."""
    mock_fn = Mock(
        side_effect=[
            Exception("Too Many Requests"),
            Exception("GatewayTimeout"),
            "success",
        ]
    )

    result = call_with_retries(mock_fn, num_retries=3, backoff_factor=2.0)

    assert result == "success"
    assert mock_fn.call_count == 3  # Called once for each try
    assert mock_sleep.call_count == 2  # Sleep called twice for retries


@patch("time.sleep")
def test_call_with_retries_failure_after_retry(mock_sleep: Mock) -> None:
    """Test retry behavior when function succeeds after retryable failures."""
    mock_fn = Mock(
        side_effect=[
            Exception("Too Many Requests"),
            Exception("GatewayTimeout"),
            "success",
        ]
    )

    with pytest.raises(Exception):
        call_with_retries(mock_fn, num_retries=2, backoff_factor=2.0)

    assert mock_fn.call_count == 2
    assert (
        mock_sleep.call_count == 1
    )  # Sleep called once after the first failure but not after the second/last failure


@patch("time.sleep")
def test_call_with_retries_direct_failure(mock_sleep: Mock) -> None:
    """Test retry behavior when function succeeds after retryable failures."""
    mock_fn = Mock(
        side_effect=[
            RuntimeError("Non Retryable Runtime Error"),
            "success",
        ]
    )

    with pytest.raises(RuntimeError):
        call_with_retries(mock_fn, num_retries=5, backoff_factor=2.0)

    assert mock_fn.call_count == 1  # Only called once, no retries
    assert mock_sleep.call_count == 0  # Sleep not called since it failed immediately


def test_dump_path(tmp_upath: UPath) -> None:
    tmp_upath = UPath(tmp_upath.resolve())

    # relative enclosed
    dataset_path = tmp_upath / "test_dataset"
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # relative not enclosed
    dataset_path = tmp_upath / "test_dataset"
    path = dataset_path / ".." / "test_dataset2" / "test.txt"
    assert (
        dump_path(path, dataset_path)
        == f"{tmp_upath.as_posix()}/test_dataset2/test.txt"
    )

    # dataset path with ..
    dataset_path = tmp_upath / "test_dataset" / ".." / "test_dataset2"
    path = tmp_upath / "test_dataset" / "test.txt"
    assert (
        dump_path(path, dataset_path) == f"{tmp_upath.as_posix()}/test_dataset/test.txt"
    )

    dataset_path = tmp_upath / "test_dataset" / ".." / "test_dataset2"
    path = tmp_upath / "test_dataset2" / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # dataset path is a prefix
    dataset_path = tmp_upath / "test_dataset"
    path = tmp_upath / "test_dataset_longer" / "test.txt"
    assert (
        dump_path(path, dataset_path)
        == f"{tmp_upath.as_posix()}/test_dataset_longer/test.txt"
    )

    # s3 non relative
    dataset_path = tmp_upath / "test_dataset"
    path = UPath(
        "s3://bucket/test.txt",
        endpoint_url="https://s3.amazonaws.com",
    )
    assert dump_path(path, dataset_path) == "s3://s3.amazonaws.com/bucket/test.txt"

    # s3 relative
    dataset_path = UPath(
        "s3://bucket/test_dataset",
        endpoint_url="https://s3.amazonaws.com",
    )
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # s3 dataset path is a prefix
    dataset_path = UPath(
        "s3://bucket/test_dataset",
        endpoint_url="https://s3.amazonaws.com",
    )
    path = (
        UPath(
            "s3://bucket/test_dataset_longer",
            endpoint_url="https://s3.amazonaws.com",
        )
        / "test.txt"
    )
    assert (
        dump_path(path, dataset_path)
        == "s3://s3.amazonaws.com/bucket/test_dataset_longer/test.txt"
    )

    # s3 with ..
    dataset_path = UPath(
        "s3://bucket/test_dataset",
        endpoint_url="https://s3.amazonaws.com",
    )
    path = dataset_path / ".." / "test_dataset2" / "test.txt"
    assert (
        dump_path(path, dataset_path)
        == "s3://s3.amazonaws.com/bucket/test_dataset2/test.txt"
    )

    path = dataset_path / ".." / "test_dataset2" / "test.txt"
    assert (
        dump_path(path, dataset_path)
        == "s3://s3.amazonaws.com/bucket/test_dataset2/test.txt"
    )

    path = dataset_path / ".." / "test_dataset" / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # s3 dataset with ..
    dataset_path = (
        UPath(
            "s3://bucket/test_dataset",
            endpoint_url="https://s3.amazonaws.com",
        )
        / ".."
        / "test_dataset2"
    )
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    path = UPath(
        "s3://bucket/test_dataset/test.txt",
        endpoint_url="https://s3.amazonaws.com",
    )
    assert (
        dump_path(path, dataset_path)
        == "s3://s3.amazonaws.com/bucket/test_dataset/test.txt"
    )

    path = (
        UPath(
            "s3://bucket/",
            endpoint_url="https://s3.amazonaws.com",
        )
        / "test_dataset2"
        / "test.txt"
    )
    assert dump_path(path, dataset_path) == "./test.txt"


def test_copytree_with_ignore() -> None:
    # use in-memory fs to demonstrate functionality with non-local UPaths
    with TestTemporaryDirectoryNonLocal() as tmp_dir:
        """
        Folder structure:
        tmp_dir/
            src/
                file1.txt
                file2.log --> X
                subdir/
                    file3.txt
                    file4.log --> X
                subdir2/ --> X
                    file5.txt
        """
        src_dir = tmp_dir / "src"
        src_dir.mkdir()

        (src_dir / "file1.txt").write_text("File 1")
        (src_dir / "file2.log").write_text("File 2")
        (src_dir / "subdir").mkdir()
        (src_dir / "subdir" / "file3.txt").write_text("File 3")
        (src_dir / "subdir" / "file4.log").write_text("File 4")
        (src_dir / "subdir2").mkdir()
        (src_dir / "subdir2" / "file5.txt").write_text("File 5")

        dst_dir = tmp_dir / "dst"

        copytree(src_dir, dst_dir, ignore=ignore_patterns("*.log", "subdir2"))

        # txt files should have been copied
        assert (dst_dir / "file1.txt").exists()
        assert (dst_dir / "subdir" / "file3.txt").exists()

        # log files should have been ignored
        assert not (dst_dir / "file2.log").exists()
        assert not (dst_dir / "subdir" / "file4.log").exists()

        # subdir2 should have been ignored entirely
        assert not (dst_dir / "subdir2").exists()

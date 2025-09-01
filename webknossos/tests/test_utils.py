from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from upath import UPath

from webknossos.utils import call_with_retries, dump_path


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


def test_dump_path(tmp_path: Path) -> None:
    tmp_path = UPath(tmp_path.resolve())

    # relative enclosed
    dataset_path = tmp_path / "test_dataset"
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # relative not enclosed
    dataset_path = tmp_path / "test_dataset"
    path = dataset_path / ".." / "test_dataset2" / "test.txt"
    assert (
        dump_path(path, dataset_path) == f"{tmp_path.as_posix()}/test_dataset2/test.txt"
    )

    # dataset path with ..
    dataset_path = tmp_path / "test_dataset" / ".." / "test_dataset2"
    path = tmp_path / "test_dataset" / "test.txt"
    assert (
        dump_path(path, dataset_path) == f"{tmp_path.as_posix()}/test_dataset/test.txt"
    )

    dataset_path = tmp_path / "test_dataset" / ".." / "test_dataset2"
    path = tmp_path / "test_dataset2" / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # dataset path is a prefix
    dataset_path = tmp_path / "test_dataset"
    path = tmp_path / "test_dataset_longer" / "test.txt"
    assert (
        dump_path(path, dataset_path)
        == f"{tmp_path.as_posix()}/test_dataset_longer/test.txt"
    )

    # s3 non relative
    dataset_path = tmp_path / "test_dataset"
    path = UPath(
        "s3://bucket/test.txt",
        client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
    )
    assert dump_path(path, dataset_path) == "s3://s3.amazonaws.com/bucket/test.txt"

    # s3 relative
    dataset_path = UPath(
        "s3://bucket/test_dataset",
        client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
    )
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    # s3 dataset path is a prefix
    dataset_path = UPath(
        "s3://bucket/test_dataset",
        client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
    )
    path = (
        UPath(
            "s3://bucket/test_dataset_longer",
            client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
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
        client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
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
            client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
        )
        / ".."
        / "test_dataset2"
    )
    path = dataset_path / "test.txt"
    assert dump_path(path, dataset_path) == "./test.txt"

    path = UPath(
        "s3://bucket/test_dataset/test.txt",
        client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
    )
    assert (
        dump_path(path, dataset_path)
        == "s3://s3.amazonaws.com/bucket/test_dataset/test.txt"
    )

    path = (
        UPath(
            "s3://bucket/",
            client_kwargs={"endpoint_url": "https://s3.amazonaws.com"},
        )
        / "test_dataset2"
        / "test.txt"
    )
    assert dump_path(path, dataset_path) == "./test.txt"

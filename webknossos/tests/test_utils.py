from unittest.mock import Mock, patch

import pytest

from webknossos.utils import call_with_retries


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

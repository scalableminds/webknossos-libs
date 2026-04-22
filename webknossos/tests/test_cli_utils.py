"""Unit tests for webknossos.cli._utils.get_executor_for_args and parse_job_resources."""

from unittest.mock import MagicMock, patch

import pytest
import typer
from cluster_tools import BatchingExecutor

from webknossos.cli._utils import (
    DistributionStrategy,
    get_executor_for_args,
    parse_job_resources,
)

# ---------------------------------------------------------------------------
# parse_job_resources
# ---------------------------------------------------------------------------


def test_parse_job_resources_key_value_format() -> None:
    assert parse_job_resources("mem=32G,time=02:00:00") == {
        "mem": "32G",
        "time": "02:00:00",
    }


def test_parse_job_resources_single_pair() -> None:
    assert parse_job_resources("mem=32G") == {"mem": "32G"}


def test_parse_job_resources_int_conversion() -> None:
    # dashes in keys are converted to underscores
    assert parse_job_resources("nice=10,mem=32G") == {
        "nice": 10,
        "mem": "32G",
    }


def test_parse_job_resources_json_backward_compat() -> None:
    assert parse_job_resources('{"mem": "32G"}') == {"mem": "32G"}


def test_parse_job_resources_json_with_int_backward_compat() -> None:
    assert parse_job_resources('{"batch_size": 100, "mem": "32G"}') == {
        "batch_size": 100,
        "mem": "32G",
    }


def test_parse_job_resources_invalid_no_equals() -> None:
    with pytest.raises(ValueError):
        parse_job_resources("badformat")


def test_parse_job_resources_invalid_json() -> None:
    with pytest.raises(ValueError):
        parse_job_resources("{bad json}")


# ---------------------------------------------------------------------------
# get_executor_for_args — error cases
# ---------------------------------------------------------------------------


def test_slurm_requires_job_resources() -> None:
    with pytest.raises(typer.BadParameter, match="--job-resources"):
        get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.SLURM,
            job_resources=None,
        )


def test_kubernetes_requires_job_resources() -> None:
    with pytest.raises(typer.BadParameter, match="--job-resources"):
        get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.KUBERNETES,
            job_resources=None,
        )


def test_slurm_batching_requires_job_resources() -> None:
    with pytest.raises(typer.BadParameter, match="--job-resources"):
        get_executor_for_args(
            jobs=10,
            distribution_strategy=DistributionStrategy.SLURM_BATCHING,
            job_resources=None,
        )


def test_slurm_batching_requires_jobs_when_no_batch_size() -> None:
    with pytest.raises(typer.BadParameter, match="--jobs"):
        get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.SLURM_BATCHING,
            job_resources={"mem": "1G"},
        )


def test_slurm_batching_rejects_jobs_with_batch_size() -> None:
    with pytest.raises(typer.BadParameter, match="--jobs"):
        get_executor_for_args(
            jobs=4,
            distribution_strategy=DistributionStrategy.SLURM_BATCHING,
            job_resources={"batch_size": 5, "mem": "1G"},
        )


# ---------------------------------------------------------------------------
# get_executor_for_args — happy paths, local executors (no mocking)
# ---------------------------------------------------------------------------


def test_multiprocessing_returns_executor_with_explicit_jobs() -> None:
    with get_executor_for_args(
        jobs=2,
        distribution_strategy=DistributionStrategy.MULTIPROCESSING,
        job_resources=None,
    ):
        pass


def test_multiprocessing_returns_executor_with_default_jobs() -> None:
    with get_executor_for_args(
        jobs=None,
        distribution_strategy=DistributionStrategy.MULTIPROCESSING,
        job_resources=None,
    ):
        pass


def test_sequential_returns_executor() -> None:
    with get_executor_for_args(
        jobs=None,
        distribution_strategy=DistributionStrategy.SEQUENTIAL,
        job_resources=None,
    ):
        pass


# ---------------------------------------------------------------------------
# get_executor_for_args — happy paths, cluster executors (mock get_executor)
# ---------------------------------------------------------------------------


def _make_mock_executor() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def test_slurm_calls_get_executor_with_correct_args() -> None:
    mock_executor = _make_mock_executor()
    with patch(
        "webknossos.cli._utils.get_executor", return_value=mock_executor
    ) as mock_get:
        with get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.SLURM,
            job_resources={"mem": "1G"},
        ):
            pass
    mock_get.assert_called_once_with(
        "slurm",
        debug=True,
        keep_logs=True,
        job_resources={"mem": "1G"},
    )


def test_kubernetes_calls_get_executor_with_correct_args() -> None:
    mock_executor = _make_mock_executor()
    with patch(
        "webknossos.cli._utils.get_executor", return_value=mock_executor
    ) as mock_get:
        with get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.KUBERNETES,
            job_resources={"memory": "1G"},
        ):
            pass
    mock_get.assert_called_once_with(
        "kubernetes",
        debug=True,
        keep_logs=True,
        job_resources={"memory": "1G"},
    )


def test_slurm_batching_uses_jobs_as_target_job_count() -> None:
    mock_executor = _make_mock_executor()
    with patch(
        "webknossos.cli._utils.get_executor", return_value=mock_executor
    ) as mock_get:
        result = get_executor_for_args(
            jobs=10,
            distribution_strategy=DistributionStrategy.SLURM_BATCHING,
            job_resources={"mem": "1G"},
        )
    assert isinstance(result, BatchingExecutor)
    mock_get.assert_called_once_with(
        "slurm",
        debug=True,
        keep_logs=True,
        job_resources={"mem": "1G"},
    )


def test_slurm_batching_with_batch_size() -> None:
    mock_executor = _make_mock_executor()
    with patch(
        "webknossos.cli._utils.get_executor", return_value=mock_executor
    ) as mock_get:
        result = get_executor_for_args(
            jobs=None,
            distribution_strategy=DistributionStrategy.SLURM_BATCHING,
            job_resources={"batch_size": 5, "mem": "1G"},
        )
    assert isinstance(result, BatchingExecutor)
    # batch_size is popped before passing resources to get_executor
    mock_get.assert_called_once_with(
        "slurm",
        debug=True,
        keep_logs=True,
        job_resources={"mem": "1G"},
    )

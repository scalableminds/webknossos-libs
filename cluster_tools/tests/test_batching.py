from pathlib import Path
from unittest.mock import patch

import pytest

import cluster_tools
from cluster_tools.executors.batching import BatchingExecutor
from cluster_tools.executors.multiprocessing_ import MultiprocessingExecutor
from cluster_tools.executors.sequential import SequentialExecutor


def double(x: int) -> int:
    return x * 2


@pytest.fixture(params=["sequential", "multiprocessing"])
def inner_executor(request: pytest.FixtureRequest) -> cluster_tools.Executor:
    if request.param == "sequential":
        return SequentialExecutor()
    return MultiprocessingExecutor(max_workers=2)


def test_map_basic(inner_executor: cluster_tools.Executor) -> None:
    with BatchingExecutor(inner_executor, batch_size=3) as executor:
        results = list(executor.map(double, [1, 2, 3, 4, 5]))
    assert results == [2, 4, 6, 8, 10]


def test_map_exact_batch_boundary() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        results = list(executor.map(double, [1, 2, 3]))
    assert results == [2, 4, 6]


def test_map_single_item() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        results = list(executor.map(double, [5]))
    assert results == [10]


def test_map_empty() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        results = list(executor.map(double, []))
    assert results == []


def test_map_calls_inner_with_batches() -> None:
    """Verify the inner executor receives batches, not individual items."""
    inner = SequentialExecutor()
    received_batches: list[list[int]] = []
    original_map = inner.map

    def spy(fn, iterables, **kwargs):  # type: ignore[no-untyped-def]
        batches = list(iterables)
        received_batches.extend(batches)
        return original_map(fn, iter(batches), **kwargs)

    with patch.object(inner, "map", side_effect=spy):
        with BatchingExecutor(inner, batch_size=3) as executor:
            results = list(executor.map(double, range(7)))

    assert results == [0, 2, 4, 6, 8, 10, 12]
    assert received_batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_map_to_futures(inner_executor: cluster_tools.Executor) -> None:
    with BatchingExecutor(inner_executor, batch_size=3) as executor:
        futures = executor.map_to_futures(double, [1, 2, 3, 4, 5])
    assert len(futures) == 5
    assert [f.result() for f in futures] == [2, 4, 6, 8, 10]


def test_map_to_futures_empty() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        futures = executor.map_to_futures(double, [])
    assert futures == []


def test_map_chunksize_raises() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        with pytest.raises(ValueError):
            list(executor.map(double, [1, 2, 3], chunksize=2))


def test_map_to_futures_output_pickle_path_getter_raises() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        with pytest.raises(NotImplementedError):
            executor.map_to_futures(
                double, [1], output_pickle_path_getter=lambda _: Path("/tmp/x")
            )

from concurrent.futures import CancelledError, Future
from pathlib import Path
from unittest.mock import patch

import pytest

import cluster_tools
from cluster_tools.executors.batching import BatchingExecutor
from cluster_tools.executors.multiprocessing_ import MultiprocessingExecutor
from cluster_tools.executors.sequential import SequentialExecutor


def double(x: int) -> int:
    return x * 2


def raise_runtime_error(_: int) -> int:
    raise RuntimeError("failure")


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


def test_map_to_futures_exception_propagates_to_all_item_futures() -> None:
    # SequentialExecutor raises synchronously, so use MultiprocessingExecutor
    # to exercise the on_batch_done exception path (failure arrives via future).
    with BatchingExecutor(
        MultiprocessingExecutor(max_workers=1), batch_size=3
    ) as executor:
        futures = executor.map_to_futures(raise_runtime_error, [1, 2, 3])

    assert len(futures) == 3
    for fut in futures:
        with pytest.raises(RuntimeError, match="failure"):
            fut.result()


def test_submit_warns() -> None:
    with BatchingExecutor(SequentialExecutor(), batch_size=3) as executor:
        with pytest.warns(UserWarning, match="bypasses batching"):
            fut = executor.submit(double, 5)
    assert fut.result() == 10


def test_init_requires_one_of_batch_size_or_target_job_count() -> None:
    with pytest.raises(ValueError, match="Either batch_size or target_job_count"):
        BatchingExecutor(SequentialExecutor())


def test_init_rejects_both() -> None:
    with pytest.raises(ValueError, match="not both"):
        BatchingExecutor(SequentialExecutor(), batch_size=3, target_job_count=2)


def test_init_rejects_zero_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        BatchingExecutor(SequentialExecutor(), batch_size=0)


def test_init_rejects_zero_target_job_count() -> None:
    with pytest.raises(ValueError, match="target_job_count must be greater than 0"):
        BatchingExecutor(SequentialExecutor(), target_job_count=0)


def test_target_job_count_map() -> None:
    """With target_job_count=2 and 7 items, batch_size=ceil(7/2)=4 → 2 jobs."""
    inner = SequentialExecutor()
    received_batches: list[list[int]] = []
    original_map = inner.map

    def spy(fn, iterables, **kwargs):  # type: ignore[no-untyped-def]
        batches = list(iterables)
        received_batches.extend(batches)
        return original_map(fn, iter(batches), **kwargs)

    with patch.object(inner, "map", side_effect=spy):
        with BatchingExecutor(inner, target_job_count=2) as executor:
            results = list(executor.map(double, range(7)))

    assert results == [0, 2, 4, 6, 8, 10, 12]
    assert received_batches == [[0, 1, 2, 3], [4, 5, 6]]


def test_target_job_count_map_to_futures() -> None:
    with BatchingExecutor(SequentialExecutor(), target_job_count=2) as executor:
        futures = executor.map_to_futures(double, [1, 2, 3, 4, 5])
    assert [f.result() for f in futures] == [2, 4, 6, 8, 10]


def test_target_job_count_empty() -> None:
    with BatchingExecutor(SequentialExecutor(), target_job_count=3) as executor:
        results = list(executor.map(double, []))
    assert results == []


def test_get_executor() -> None:
    executor = cluster_tools.get_executor(
        "batching",
        executor={
            "name": "multiprocessing",
            "max_workers": 3,
        },
        batch_size=10,
    )
    assert isinstance(executor, BatchingExecutor)
    assert isinstance(executor._executor, MultiprocessingExecutor)
    assert executor._executor._max_workers == 3  # type: ignore[attr-defined]
    assert executor.batch_size == 10


def test_get_executor_target_job_count() -> None:
    executor = cluster_tools.get_executor(
        "batching",
        executor={"name": "sequential"},
        target_job_count=4,
    )
    assert isinstance(executor, BatchingExecutor)
    assert executor.target_job_count == 4


def test_get_executor_missing_sizing_raises() -> None:
    with pytest.raises(ValueError, match="batch_size.*target_job_count"):
        cluster_tools.get_executor(
            "batching",
            executor={"name": "sequential"},
        )

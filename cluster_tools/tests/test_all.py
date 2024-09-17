import logging
import os
import tempfile
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import pytest

if TYPE_CHECKING:
    from distributed import LocalCluster

import cluster_tools


# "Worker" functions.
def square(n: float) -> float:
    return n * n


def sleep(duration: float) -> float:
    time.sleep(duration)
    return duration


logging.basicConfig()

_dask_cluster: Optional["LocalCluster"] = None


def raise_if(msg: str, _bool: bool) -> None:
    if _bool:
        raise Exception("raise_if was called with True: {}".format(msg))


def get_executors(
    with_pickling: bool = False,
) -> List[cluster_tools.Executor]:
    global _dask_cluster
    executor_keys = {
        "slurm",
        "kubernetes",
        "dask",
        "multiprocessing",
        "sequential",
    }
    if with_pickling:
        executor_keys.add("multiprocessing_with_pickling")
        executor_keys.add("sequential_with_pickling")

    if "PYTEST_EXECUTORS" in os.environ:
        executor_keys = executor_keys.intersection(
            os.environ["PYTEST_EXECUTORS"].split(",")
        )

    executors: List[cluster_tools.Executor] = []
    if "slurm" in executor_keys:
        executors.append(
            cluster_tools.get_executor(
                "slurm", debug=True, job_resources={"mem": "100M"}
            )
        )
    if "kubernetes" in executor_keys:
        executors.append(
            cluster_tools.get_executor(
                "kubernetes",
                debug=True,
                job_resources={
                    "memory": "1G",
                    "image": "scalableminds/cluster-tools:latest",
                },
            )
        )
    if "multiprocessing" in executor_keys:
        executors.append(cluster_tools.get_executor("multiprocessing", max_workers=5))
    if "sequential" in executor_keys:
        executors.append(cluster_tools.get_executor("sequential"))
    if "dask" in executor_keys:
        if not _dask_cluster:
            from distributed import LocalCluster, Worker

            _dask_cluster = LocalCluster(
                worker_class=Worker, resources={"mem": 20e9, "cpus": 4}, nthreads=6
            )
        executors.append(
            cluster_tools.get_executor("dask", job_resources={"address": _dask_cluster})
        )
    if "multiprocessing_with_pickling" in executor_keys:
        executors.append(cluster_tools.get_executor("multiprocessing_with_pickling"))
    if "pbs" in executor_keys:
        executors.append(cluster_tools.get_executor("pbs"))
    if "sequential_with_pickling" in executor_keys:
        executors.append(cluster_tools.get_executor("sequential_with_pickling"))
    return executors


@pytest.mark.skip(
    reason="The test is flaky on the CI for some reason. Disable it for now."
)
def test_uncaught_warning() -> None:
    """
    This test ensures that there are warnings for "uncaught" futures.
    """

    # Log to a specific file which we can check for
    log_file_name = "warning.log"
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)

    cases = [False, True]

    def expect_marker(marker: str, msg: str, should_exist: bool = True) -> None:
        def maybe_negate(b: bool) -> bool:
            return b if should_exist else not b

        fh.flush()
        with open(log_file_name) as file:
            content = file.read()
            assert maybe_negate(marker in content), msg

    # In the following 4 cases we check whether there is a/no warning when using
    # map/submit with/without checking the futures.
    for exc in get_executors():
        marker = "map-expect-warning"
        with exc:
            exc.map(partial(raise_if, marker), cases)
        expect_marker(marker, "There should be a warning for an uncaught Future in map")

    for exc in get_executors():
        marker = "map-dont-expect-warning"
        with exc:
            try:
                list(exc.map(partial(raise_if, marker), cases))
            except Exception:
                pass
        expect_marker(
            marker, "There should be no warning for an uncaught Future in map", False
        )

    for exc in get_executors():
        marker = "submit-expect-warning"
        with exc:
            futures = [exc.submit(partial(raise_if, marker), b) for b in cases]
        expect_marker(
            marker, "There should be no warning for an uncaught Future in submit"
        )

    for exc in get_executors():
        marker = "submit-dont-expect-warning"
        with exc:
            futures = [exc.submit(partial(raise_if, marker), b) for b in cases]
            try:
                for f in futures:
                    f.result()
            except Exception:
                pass
        expect_marker(
            marker, "There should be a warning for an uncaught Future in submit", False
        )

    logger.removeHandler(fh)


@pytest.mark.parametrize("exc", get_executors(with_pickling=True), ids=type)
def test_submit(exc: cluster_tools.Executor) -> None:
    def run_square_numbers(executor: cluster_tools.Executor) -> None:
        with executor:
            job_count = 3
            job_range = range(job_count)
            futures = [executor.submit(square, n) for n in job_range]
            for future, job_index in zip(futures, job_range):
                assert future.result() == square(job_index)

    run_square_numbers(exc)


def get_pid() -> int:
    return os.getpid()


@pytest.mark.parametrize("exc", get_executors(with_pickling=True), ids=type)
def test_process_id(exc: cluster_tools.Executor) -> None:
    outer_pid = os.getpid()

    def compare_pids(executor: cluster_tools.Executor) -> None:
        with executor:
            future = executor.submit(get_pid)
            inner_pid = future.result()

            should_differ = not isinstance(
                executor,
                (
                    cluster_tools.SequentialExecutor,
                    cluster_tools.SequentialPickleExecutor,
                ),
            )

            if should_differ:
                assert (
                    inner_pid != outer_pid
                ), f"Inner and outer pid should differ, but both are {inner_pid}."
            else:
                assert (
                    inner_pid == outer_pid
                ), f"Inner and outer pid should be equal, but {inner_pid} != {outer_pid}."

    compare_pids(exc)


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_unordered_sleep(exc: cluster_tools.Executor) -> None:
    with exc:
        durations = [3, 1]
        futures = [exc.submit(sleep, n) for n in durations]
        results = [f.result() for f in exc.as_completed(futures)]

        if isinstance(
            exc,
            (
                cluster_tools.SequentialExecutor,
                cluster_tools.SequentialPickleExecutor,
            ),
        ):
            # futures.as_completed does not return previously completed futures in completion order.
            # For sequential executors as_completed is only called after all futures completed, though.
            results.sort()

        durations.sort()
        for duration, result in zip(durations, results):
            assert result == duration


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_map_to_futures(exc: cluster_tools.Executor) -> None:
    with exc:
        durations = [3, 1]
        futures = exc.map_to_futures(sleep, durations)
        results = [f.result() for f in exc.as_completed(futures)]

        if isinstance(
            exc,
            (cluster_tools.SequentialExecutor, cluster_tools.SequentialPickleExecutor),
        ):
            # futures.as_completed does not return previously completed futures in completion order.
            # For sequential executors as_completed is only called after all futures completed, though.
            results.sort()

        durations.sort()
        for duration, result in zip(durations, results):
            assert result == duration


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_empty_map_to_futures(exc: cluster_tools.Executor) -> None:
    with exc:
        futures = exc.map_to_futures(sleep, [])
        results = [f.result() for f in futures]
        assert len(results) == 0


def output_pickle_path_getter(tmp_dir: str, chunk: int) -> Path:
    return Path(tmp_dir) / f"test_{chunk}.pickle"


@pytest.mark.parametrize("exc", get_executors(with_pickling=True), ids=type)
def test_map_to_futures_with_pickle_paths(exc: cluster_tools.Executor) -> None:
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        with exc:
            numbers = [2, 1]
            futures = exc.map_to_futures(
                square,
                numbers,
                output_pickle_path_getter=partial(output_pickle_path_getter, tmp_dir),
            )
            results = [f.result() for f in exc.as_completed(futures)]
            assert set(results) == {1, 4}

        for number in numbers:
            assert Path(
                output_pickle_path_getter(tmp_dir, number)
            ).exists(), f"File for chunk {number} should exist."


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_submit_with_pickle_paths(exc: cluster_tools.Executor) -> None:
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:

        def run_square_numbers(executor: cluster_tools.Executor) -> Path:
            with executor:
                job_count = 3
                job_range = range(job_count)

                futures = []
                for n in job_range:
                    output_path = Path(tmp_dir) / f"{n}.pickle"
                    cfut_options = {"output_pickle_path": output_path}
                    futures.append(
                        executor.submit(square, n, __cfut_options=cfut_options)  # type: ignore[call-arg]
                    )

                for future, job_index in zip(futures, job_range):
                    assert future.result() == square(job_index)
                return output_path

        output_path = run_square_numbers(exc)
        assert output_path.exists(), "Output pickle file should exist."


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_map(exc: cluster_tools.Executor) -> None:
    def run_map(executor: cluster_tools.Executor) -> None:
        with executor:
            result = list(executor.map(square, [2, 3, 4]))
            assert result == [4, 9, 16]

    run_map(exc)


@pytest.mark.parametrize("exc", get_executors(), ids=type)
def test_map_lazy(exc: cluster_tools.Executor) -> None:
    def run_map(executor: cluster_tools.Executor) -> None:
        with executor:
            result = executor.map(square, [2, 3, 4])
        assert list(result) == [4, 9, 16]

    if not isinstance(exc, cluster_tools.DaskExecutor):
        run_map(exc)


def test_executor_args() -> None:
    def pass_with(exc: cluster_tools.Executor) -> None:
        with exc:
            pass

    pass_with(cluster_tools.get_executor("sequential", non_existent_arg=True))
    # Test should succeed if the above lines don't raise an exception


class DummyEnum(Enum):
    BANANA = 0
    APPLE = 1
    PEAR = 2


def enum_consumer(value: DummyEnum) -> DummyEnum:
    assert value == DummyEnum.BANANA
    return value


@pytest.mark.parametrize(
    "executor_key", ["multiprocessing_with_pickling", "sequential_with_pickling"]
)
def test_pickling(
    executor_key: Union[
        Literal["multiprocessing_with_pickling"], Literal["sequential_with_pickling"]
    ],
) -> None:
    with cluster_tools.get_executor(executor_key) as executor:
        future = executor.submit(enum_consumer, DummyEnum.BANANA)
        assert future.result() == DummyEnum.BANANA


def test_map_to_futures_with_sequential() -> None:
    with cluster_tools.get_executor("sequential") as exc:
        durations = [3, 1]
        futures = exc.map_to_futures(sleep, durations)

        for fut in futures:
            assert (
                fut.done()
            ), "Future should immediately be finished after map_to_futures has returned"

        results = [f.result() for f in futures]

        for duration, result in zip(durations, results):
            assert result == duration

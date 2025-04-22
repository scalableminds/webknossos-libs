import logging
import os
import tempfile
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import pytest

import cluster_tools.executors
import cluster_tools.schedulers

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
        raise Exception(f"raise_if was called with True: {msg}")


# Most of the specs in this module should be executed with multiple executors. Some tests
# should be called with all executors (including the pickling variants) and some with a subset (i.e., without the pickling variants).
# In order to avoid redundant parameterization of each test, pytest_generate_tests is defined here.
# If a spec uses an `exc_with_pickling` fixture (defined as a function parameter), that test is automatically parameterized with all executors. Analogous, parameterization happens with `exc`.
# Regarding how this works in details: This function is called for each test and has access to the fixtures supplied
# to the test and most importantly can parametrize those fixtures.
def pytest_generate_tests(metafunc: Any) -> None:
    if "exc" in metafunc.fixturenames or "exc_with_pickling" in metafunc.fixturenames:
        with_pickling = "exc_with_pickling" in metafunc.fixturenames
        executor_keys = get_executor_keys(with_pickling)
        metafunc.parametrize(
            "exc_with_pickling" if with_pickling else "exc",
            executor_keys,
            indirect=True,
        )


@pytest.fixture
def exc(
    request: Any,
) -> cluster_tools.Executor:
    return get_executor(request.param)


@pytest.fixture
def exc_with_pickling(
    request: Any,
) -> cluster_tools.Executor:
    return get_executor(request.param)


def get_executor_keys(with_pickling: bool = False) -> set[str]:
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

    return executor_keys


def get_executor(environment: str) -> cluster_tools.Executor:
    global _dask_cluster

    if environment == "slurm":
        return cluster_tools.get_executor(
            "slurm", debug=True, job_resources={"mem": "100M"}
        )
    if environment == "kubernetes":
        return cluster_tools.get_executor(
            "kubernetes",
            debug=True,
            job_resources={
                "memory": "1G",
                "image": "scalableminds/cluster-tools:latest",
            },
        )
    if environment == "multiprocessing":
        return cluster_tools.get_executor("multiprocessing", max_workers=5)
    if environment == "sequential":
        return cluster_tools.get_executor("sequential")
    if environment == "dask":
        if not _dask_cluster:
            from distributed import LocalCluster, Worker

            _dask_cluster = LocalCluster(
                worker_class=Worker, resources={"mem": 20e9, "cpus": 4}, nthreads=6
            )
        return cluster_tools.get_executor(
            "dask", job_resources={"address": _dask_cluster}
        )
    if environment == "multiprocessing_with_pickling":
        return cluster_tools.get_executor("multiprocessing_with_pickling")
    if environment == "pbs":
        return cluster_tools.get_executor("pbs")
    if environment == "sequential_with_pickling":
        return cluster_tools.get_executor("sequential_with_pickling")
    raise RuntimeError("No executor specified.")


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
    for exc_key in get_executor_keys():
        exc = get_executor(exc_key)
        marker = "map-expect-warning"
        with exc:
            exc.map(partial(raise_if, marker), cases)
        expect_marker(marker, "There should be a warning for an uncaught Future in map")

    for exc_key in get_executor_keys():
        exc = get_executor(exc_key)
        marker = "map-dont-expect-warning"
        with exc:
            try:
                list(exc.map(partial(raise_if, marker), cases))
            except Exception:
                pass
        expect_marker(
            marker, "There should be no warning for an uncaught Future in map", False
        )

    for exc_key in get_executor_keys():
        exc = get_executor(exc_key)
        marker = "submit-expect-warning"
        with exc:
            futures = [exc.submit(partial(raise_if, marker), b) for b in cases]
        expect_marker(
            marker, "There should be no warning for an uncaught Future in submit"
        )

    for exc_key in get_executor_keys():
        exc = get_executor(exc_key)
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


def test_submit(exc_with_pickling: cluster_tools.Executor) -> None:
    exc = exc_with_pickling
    with exc:
        job_count = 3
        job_range = range(job_count)
        futures = [exc.submit(square, n) for n in job_range]
        for future, job_index in zip(futures, job_range):
            assert future.result() == square(job_index)


def get_pid() -> int:
    return os.getpid()


def test_process_id(exc_with_pickling: cluster_tools.Executor) -> None:
    exc = exc_with_pickling
    outer_pid = os.getpid()

    with exc:
        future = exc.submit(get_pid)
        inner_pid = future.result()

        should_differ = not isinstance(
            exc,
            cluster_tools.SequentialExecutor | cluster_tools.SequentialPickleExecutor,
        )

        if should_differ:
            assert inner_pid != outer_pid, (
                f"Inner and outer pid should differ, but both are {inner_pid}."
            )
        else:
            assert inner_pid == outer_pid, (
                f"Inner and outer pid should be equal, but {inner_pid} != {outer_pid}."
            )


def test_unordered_sleep(exc: cluster_tools.Executor) -> None:
    is_async = not isinstance(
        exc, cluster_tools.SequentialExecutor | cluster_tools.SequentialPickleExecutor
    )

    with exc:
        durations = [5, 0]
        # Slurm can be a bit slow to start up, so we need to increase the sleep time
        if isinstance(exc, cluster_tools.SlurmExecutor):
            durations = [20, 0]
        futures = [exc.submit(sleep, n) for n in durations]
        # For synchronous executors, the futures should be completed after submit returns.
        # .as_completed() would return them in reverse order in that case.
        completed_futures = exc.as_completed(futures) if is_async else futures
        results = [f.result() for f in completed_futures]

        if is_async:
            # For asynchronous executors, the jobs that sleep less should complete first
            durations.sort()

        assert durations == results


def test_map_to_futures(exc: cluster_tools.Executor) -> None:
    is_async = not isinstance(
        exc, cluster_tools.SequentialExecutor | cluster_tools.SequentialPickleExecutor
    )

    with exc:
        durations = [5, 0]
        futures = exc.map_to_futures(sleep, durations)
        # For synchronous executors, the futures should be completed after submit returns.
        # .as_completed() would return them in reverse order in that case.
        completed_futures = exc.as_completed(futures) if is_async else futures
        results = [f.result() for f in completed_futures]

        if is_async:
            # For asynchronous executors, the jobs that sleep less should complete first
            durations.sort()

        assert durations == results


def test_empty_map_to_futures(exc: cluster_tools.Executor) -> None:
    with exc:
        futures = exc.map_to_futures(sleep, [])
        results = [f.result() for f in futures]
        assert len(results) == 0


def output_pickle_path_getter(tmp_dir: str, chunk: int) -> Path:
    return Path(tmp_dir) / f"test_{chunk}.pickle"


def test_map_to_futures_with_pickle_paths(
    exc_with_pickling: cluster_tools.Executor,
) -> None:
    exc = exc_with_pickling
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
            assert Path(output_pickle_path_getter(tmp_dir, number)).exists(), (
                f"File for chunk {number} should exist."
            )


def test_submit_with_pickle_paths(exc: cluster_tools.Executor) -> None:
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        with exc:
            job_count = 3
            job_range = range(job_count)

            futures = []
            for n in job_range:
                output_path = Path(tmp_dir) / f"{n}.pickle"
                cfut_options = {"output_pickle_path": output_path}
                futures.append(
                    exc.submit(square, n, __cfut_options=cfut_options)  # type: ignore[call-arg]
                )

            for future, job_index in zip(futures, job_range):
                assert future.result() == square(job_index)

        assert output_path.exists(), "Output pickle file should exist."


def test_map(exc: cluster_tools.Executor) -> None:
    with exc:
        result = list(exc.map(square, [2, 3, 4]))
        assert result == [4, 9, 16]


def test_map_lazy(exc: cluster_tools.Executor) -> None:
    if not isinstance(exc, cluster_tools.DaskExecutor):
        with exc:
            result = exc.map(square, [2, 3, 4])
        assert list(result) == [4, 9, 16]


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
    executor_key: Literal["multiprocessing_with_pickling"]
    | Literal["sequential_with_pickling"],
) -> None:
    with cluster_tools.get_executor(executor_key) as executor:
        future = executor.submit(enum_consumer, DummyEnum.BANANA)
        assert future.result() == DummyEnum.BANANA


def test_map_to_futures_with_sequential() -> None:
    with cluster_tools.get_executor("sequential") as exc:
        durations = [1, 0]
        futures = exc.map_to_futures(sleep, durations)

        for fut in futures:
            assert fut.done(), (
                "Future should immediately be finished after map_to_futures has returned"
            )

        results = [f.result() for f in futures]

        for duration, result in zip(durations, results):
            assert result == duration

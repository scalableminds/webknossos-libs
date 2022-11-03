import concurrent.futures
import logging
import os
import tempfile
import time
from enum import Enum
from functools import partial
from pathlib import Path

import pytest

import cluster_tools


# "Worker" functions.
def square(n):
    return n * n


def sleep(duration):
    time.sleep(duration)
    return duration


logging.basicConfig()


def raise_if(msg, bool):
    if bool:
        raise Exception("raise_if was called with True: {}".format(msg))


def get_executors(with_debug_sequential=False):
    executor_keys = {
        "slurm",
        "kubernetes",
        "multiprocessing",
        "sequential",
        "test_pickling",
    }
    if with_debug_sequential:
        executor_keys.add("debug_sequential")
        executor_keys.add("sequential_new_process")

    if "PYTEST_EXECUTORS" in os.environ:
        executor_keys = executor_keys.intersection(
            os.environ["PYTEST_EXECUTORS"].split(",")
        )

    executors = []
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
    if "test_pickling" in executor_keys:
        executors.append(cluster_tools.get_executor("test_pickling"))
    if "pbs" in executor_keys:
        executors.append(cluster_tools.get_executor("pbs"))
    if "debug_sequential" in executor_keys:
        executors.append(cluster_tools.get_executor("debug_sequential"))
    return executors


@pytest.mark.skip(
    reason="The test is flaky on the CI for some reason. Disable it for now."
)
def test_uncaught_warning():
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

    def expect_marker(marker, msg, should_exist=True):
        maybe_negate = lambda b: b if should_exist else not b

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


def test_submit():
    def run_square_numbers(executor):
        with executor:
            job_count = 3
            job_range = range(job_count)
            futures = [executor.submit(square, n) for n in job_range]
            for future, job_index in zip(futures, job_range):
                assert future.result() == square(job_index)

    for exc in get_executors(with_debug_sequential=True):
        run_square_numbers(exc)


def get_pid():

    return os.getpid()


def test_process_id():

    outer_pid = os.getpid()

    def compare_pids(executor):
        with executor:
            future = executor.submit(get_pid)
            inner_pid = future.result()

            should_differ = not isinstance(exc, cluster_tools.DebugSequentialExecutor)

            if should_differ:
                assert (
                    inner_pid != outer_pid
                ), f"Inner and outer pid should differ, but both are {inner_pid}."
            else:
                assert (
                    inner_pid == outer_pid
                ), f"Inner and outer pid should be equal, but {inner_pid} != {outer_pid}."

    for exc in get_executors(with_debug_sequential=True):
        compare_pids(exc)


def test_unordered_sleep():
    for exc in get_executors():
        with exc:
            durations = [10, 5]
            futures = [exc.submit(sleep, n) for n in durations]
            if not isinstance(exc, cluster_tools.SequentialExecutor):
                durations.sort()
            for duration, future in zip(
                durations, concurrent.futures.as_completed(futures)
            ):
                assert future.result() == duration


def test_unordered_map():
    for exc in get_executors():
        with exc:
            durations = [15, 1]
            results_gen = exc.map_unordered(sleep, durations)
            results = list(results_gen)

            if not isinstance(exc, cluster_tools.SequentialExecutor):
                durations.sort()

            for duration, result in zip(durations, results):
                assert result == duration


def test_map_to_futures():
    for exc in get_executors():
        with exc:
            durations = [15, 1]
            futures = exc.map_to_futures(sleep, durations)
            results = []

            for i, duration in enumerate(concurrent.futures.as_completed(futures)):
                results.append(duration.result())

            if not isinstance(exc, cluster_tools.SequentialExecutor):
                durations.sort()

            for duration, result in zip(durations, results):
                assert result == duration


def test_empty_map_to_futures():
    for exc in get_executors():
        with exc:
            futures = exc.map_to_futures(sleep, [])
            results = [f.result() for f in futures]
            assert len(results) == 0


def output_pickle_path_getter(tmp_dir, chunk):

    return Path(tmp_dir) / f"test_{chunk}.pickle"


def test_map_to_futures_with_pickle_paths():

    for exc in get_executors(with_debug_sequential=True):
        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            with exc:
                durations = [2, 1]
                futures = exc.map_to_futures(
                    sleep,
                    durations,
                    output_pickle_path_getter=partial(
                        output_pickle_path_getter, tmp_dir
                    ),
                )
                results = []

                for i, duration in enumerate(concurrent.futures.as_completed(futures)):
                    results.append(duration.result())

                assert 2 in results
                assert 1 in results

            for duration in durations:
                assert Path(
                    output_pickle_path_getter(tmp_dir, duration)
                ).exists(), f"File for chunk {duration} should exist."


def test_submit_with_pickle_paths():
    for (idx, exc) in enumerate(get_executors()):
        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:

            def run_square_numbers(idx, executor):
                with executor:
                    job_count = 3
                    job_range = range(job_count)

                    futures = []
                    for n in job_range:
                        output_path = Path(tmp_dir) / f"{idx}_{n}.pickle"
                        cfut_options = {"output_pickle_path": output_path}
                        futures.append(
                            executor.submit(square, n, __cfut_options=cfut_options)
                        )

                    for future, job_index in zip(futures, job_range):
                        assert future.result() == square(job_index)
                    return output_path

            output_path = run_square_numbers(idx, exc)
            assert output_path.exists(), "Output pickle file should exist."


def test_map():
    def run_map(executor):
        with executor:
            result = list(executor.map(square, [2, 3, 4]))
            assert result == [4, 9, 16]

    for exc in get_executors():
        run_map(exc)


def test_map_lazy():
    def run_map(executor):
        with executor:
            result = executor.map(square, [2, 3, 4])
        assert list(result) == [4, 9, 16]

    for exc in get_executors():
        run_map(exc)


def test_executor_args():
    def pass_with(exc):
        with exc:
            pass

    pass_with(cluster_tools.get_executor("sequential", non_existent_arg=True))
    # Test should succeed if the above lines don't raise an exception


class DummyEnum(Enum):
    BANANA = 0
    APPLE = 1
    PEAR = 2


def enum_consumer(value):
    assert value == DummyEnum.BANANA


def test_cloudpickle_serialization():
    enum_consumer_inner = enum_consumer

    for fn in [enum_consumer, enum_consumer_inner]:
        try:
            with cluster_tools.get_executor("test_pickling") as executor:
                fut = executor.submit(fn, DummyEnum.BANANA)
            assert fn == enum_consumer
        except Exception:
            assert fn != enum_consumer

    assert True


def test_map_to_futures_with_debug_sequential():

    with cluster_tools.get_executor("debug_sequential") as exc:
        durations = [4, 1]
        futures = exc.map_to_futures(sleep, durations)

        for fut in futures:
            assert (
                fut.done()
            ), "Future should immediately be finished after map_to_futures has returned"

        results = []
        for i, duration in enumerate(futures):
            results.append(duration.result())

        for duration, result in zip(durations, results):
            assert result == duration

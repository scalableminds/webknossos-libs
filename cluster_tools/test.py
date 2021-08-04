import cluster_tools
import concurrent.futures
import time
import logging
from enum import Enum
from functools import partial
import os
import pytest
import shutil
import contextlib
import io
import multiprocessing as mp
from pathlib import Path
import tempfile

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
    executors = [
        cluster_tools.get_executor("slurm", debug=True, job_resources={"mem": "100M"}),
        cluster_tools.get_executor("multiprocessing", max_workers=5),
        cluster_tools.get_executor("sequential"),
        cluster_tools.get_executor("test_pickling"),
        # cluster_tools.get_executor("pbs"),
    ]

    if with_debug_sequential:
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
    """Get host identifying information about the servers running
    our jobs.
    """
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


def expect_fork():
    assert mp.get_start_method() == "fork"
    return True


def expect_spawn():
    assert mp.get_start_method() == "spawn"
    return True


def test_map_with_spawn():

    with cluster_tools.get_executor("multiprocessing", max_workers=5) as executor:
        assert executor.submit(
            expect_fork
        ).result(), "Multiprocessing should use fork by default"

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method=None
    ) as executor:
        assert executor.submit(
            expect_fork
        ).result(), "Multiprocessing should use fork if start_method is None"

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method="spawn"
    ) as executor:
        assert executor.submit(
            expect_spawn
        ).result(), "Multiprocessing should use spawn if requested"

    with cluster_tools.get_executor(
        "slurm", max_workers=5, start_method="spawn"
    ) as executor:
        assert executor.submit(
            expect_fork
        ).result(), "Slurm should ignore provided start_method"


def test_map_lazy():
    def run_map(executor):
        with executor:
            result = executor.map(square, [2, 3, 4])
        assert list(result) == [4, 9, 16]

    for exc in get_executors():
        run_map(exc)


def test_slurm_submit_returns_job_ids():
    exc = cluster_tools.get_executor("slurm", debug=True)
    with exc:
        future = exc.submit(square, 2)
        assert isinstance(future.cluster_jobid, int)
        assert future.cluster_jobid > 0
        assert future.result() == 4


def test_slurm_cfut_dir():
    cfut_dir = "./test_cfut_dir"
    if os.path.exists(cfut_dir):
        shutil.rmtree(cfut_dir)

    exc = cluster_tools.get_executor("slurm", debug=True, cfut_dir=cfut_dir)
    with exc:
        future = exc.submit(square, 2)
        assert future.result() == 4

    assert os.path.exists(cfut_dir)
    assert len(os.listdir(cfut_dir)) == 2


def test_executor_args():
    def pass_with(exc):
        with exc:
            pass

    pass_with(
        cluster_tools.get_executor(
            "slurm", job_resources={"mem": "10M"}, non_existent_arg=True
        )
    )
    pass_with(cluster_tools.get_executor("multiprocessing", non_existent_arg=True))
    pass_with(cluster_tools.get_executor("sequential", non_existent_arg=True))

    # Test should succeed if the above lines don't raise an exception


test_output_str = "Test-Output"


def log(string):
    logging.debug(string)


def test_pickled_logging():
    def execute_with_log_level(log_level):
        logging_config = {"level": log_level}
        with cluster_tools.get_executor(
            "slurm",
            debug=True,
            job_resources={"mem": "10M"},
            logging_config=logging_config,
        ) as executor:
            fut = executor.submit(log, test_output_str)
            fut.result()

            output = ".cfut/slurmpy.{}.log.stdout".format(fut.cluster_jobid)

            with open(output, "r") as file:
                return file.read()

    debug_out = execute_with_log_level(logging.DEBUG)
    assert test_output_str in debug_out

    info_out = execute_with_log_level(logging.INFO)
    assert not (test_output_str in info_out)


def test_tailed_logging():

    with cluster_tools.get_executor(
        "slurm",
        debug=True,
        job_resources={"mem": "10M"},
        logging_config={"level": logging.DEBUG},
    ) as executor:
        secret_string = "secret_string"

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            fut = executor.submit(log, secret_string)
            executor.forward_log(fut)

        assert secret_string in f.getvalue()
        assert "jid" in f.getvalue()


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


class TestClass:
    pass


def deref_fun_helper(obj):
    clss, inst, one, two = obj
    assert one == 1
    assert two == 2
    assert isinstance(inst, clss)


def test_dereferencing_main():
    with cluster_tools.get_executor(
        "slurm", debug=True, job_resources={"mem": "10M"}
    ) as executor:
        fut = executor.submit(deref_fun_helper, (TestClass, TestClass(), 1, 2))
        fut.result()
        futs = executor.map_to_futures(
            deref_fun_helper, [(TestClass, TestClass(), 1, 2)]
        )
        futs[0].result()


def accept_high_mem(data):
    return len(data)


@pytest.mark.skip(
    reason="This test is does not pass on the CI. Probably because the machine does not have enough RAM."
)
def test_high_ram_usage():
    very_long_string = " " * 10 ** 6 * 2500

    os.environ["MULTIPROCESSING_VIA_IO"] = "True"

    with cluster_tools.get_executor("multiprocessing") as executor:
        fut1 = executor.submit(
            accept_high_mem,
            very_long_string,
            __cfut_options={"output_pickle_path": "/tmp/test.pickle"},
        )
        assert fut1.result() == len(very_long_string)

        os.environ["MULTIPROCESSING_VIA_IO_TMP_DIR"] = "."
        fut2 = executor.submit(accept_high_mem, very_long_string)
        assert fut2.result() == len(very_long_string)

    del os.environ["MULTIPROCESSING_VIA_IO"]


def fail(val):
    raise Exception("Fail()")


def test_preliminary_file_submit():

    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        output_pickle_path = Path(tmp_dir) / "test.pickle"
        preliminary_output_path = Path(tmp_dir) / "test.pickle.preliminary"

        with cluster_tools.get_executor(
            "slurm", debug=True, job_resources={"mem": "10M"}
        ) as executor:
            # Schedule failing job and verify that only a preliminary output exists
            fut = executor.submit(
                partial(fail, None),
                __cfut_options={"output_pickle_path": str(output_pickle_path)},
            )
            with pytest.raises(Exception):
                fut.result()
            assert (
                preliminary_output_path.exists()
            ), "Preliminary output file should exist"
            assert not output_pickle_path.exists(), "Final output file should not exist"

            # Schedule succeeding job with same output path
            fut = executor.submit(
                square,
                3,
                __cfut_options={"output_pickle_path": str(output_pickle_path)},
            )
            assert fut.result() == 9
            assert output_pickle_path.exists(), "Final output file should exist"
            assert (
                not preliminary_output_path.exists()
            ), "Preliminary output file should not exist anymore"


def test_preliminary_file_map():

    a_range = range(1, 4)

    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        with cluster_tools.get_executor(
            "slurm", debug=True, job_resources={"mem": "10M"}
        ) as executor:
            # Schedule failing jobs and verify that only a preliminary output exists

            futs = executor.map_to_futures(
                fail,
                list(a_range),
                output_pickle_path_getter=partial(output_pickle_path_getter, tmp_dir),
            )
            with pytest.raises(Exception):
                [fut.result() for fut in futs]

            for idx in a_range:
                output_pickle_path = Path(output_pickle_path_getter(tmp_dir, idx))
                preliminary_output_path = Path(f"{output_pickle_path}.preliminary")

                assert (
                    preliminary_output_path.exists()
                ), "Preliminary output file should exist"
                assert (
                    not output_pickle_path.exists()
                ), "Final output file should not exist"

            # Schedule succeeding jobs with same output paths
            futs = executor.map_to_futures(
                square,
                list(a_range),
                output_pickle_path_getter=partial(output_pickle_path_getter, tmp_dir),
            )
            for (fut, job_index) in zip(futs, a_range):
                fut.result() == square(job_index)

            for idx in a_range:
                output_pickle_path = Path(output_pickle_path_getter(tmp_dir, idx))
                preliminary_output_path = Path(f"{output_pickle_path}.preliminary")
                assert output_pickle_path.exists(), "Final output file should exist"
                assert (
                    not preliminary_output_path.exists()
                ), "Preliminary output file should not exist anymore"


if __name__ == "__main__":
    # Validate that slurm_executor.submit also works when being called from a __main__ module
    test_dereferencing_main()

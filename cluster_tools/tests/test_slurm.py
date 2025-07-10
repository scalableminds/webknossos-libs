import concurrent.futures
import contextlib
import gc
import io
import logging
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import pytest

import cluster_tools
from cluster_tools._utils.call import call, chcall


# "Worker" functions.
def square(n: float) -> float:
    return n * n


def sleep(duration: float) -> float:
    time.sleep(duration)
    return duration


def allocate(duration: float, num_bytes: int) -> int:
    data = b"\x00" * num_bytes
    time.sleep(duration)
    return sys.getsizeof(data)


logging.basicConfig()


def expect_fork() -> bool:
    assert mp.get_start_method() == "fork"
    return True


def test_map_with_spawn() -> None:
    with cluster_tools.get_executor(
        "slurm", max_workers=5, start_method="spawn"
    ) as executor:
        assert executor.submit(expect_fork).result(), (
            "Slurm should ignore provided start_method"
        )


def test_slurm_submit_returns_job_ids() -> None:
    exc = cluster_tools.get_executor("slurm", debug=True)
    with exc:
        future = exc.submit(square, 2)
        assert isinstance(future.cluster_jobid, str)  # type: ignore[attr-defined]
        assert int(future.cluster_jobid) > 0  # type: ignore[attr-defined]
        assert future.result() == 4


def test_slurm_cfut_dir() -> None:
    cfut_dir = "./test_cfut_dir"
    if os.path.exists(cfut_dir):
        shutil.rmtree(cfut_dir)

    exc = cluster_tools.get_executor("slurm", debug=True, cfut_dir=cfut_dir)  # type: ignore[attr-defined]
    with exc:
        future = exc.submit(square, 2)
        assert future.result() == 4

    assert os.path.exists(cfut_dir)
    assert len(os.listdir(cfut_dir)) == 1  # only the log file should still exist


def test_slurm_max_submit_user() -> None:
    max_submit_jobs = 6

    # MaxSubmitJobs can either be defined at the user or at the qos level
    for command in ["user root", "qos normal"]:
        executor = cluster_tools.get_executor("slurm", debug=True)
        original_max_submit_jobs = executor.get_max_submit_jobs()

        try:
            chcall(
                f"echo y | sacctmgr modify {command} set MaxSubmitJobs={max_submit_jobs}"
            )

            new_max_submit_jobs = executor.get_max_submit_jobs()
            assert new_max_submit_jobs == max_submit_jobs

            with executor:
                futures = executor.map_to_futures(square, range(10))

                result = [fut.result() for fut in futures]
                assert result == [i**2 for i in range(10)]

                job_ids = {fut.cluster_jobid for fut in futures}  # type: ignore[attr-defined]
                # The 10 work packages should have been scheduled as 2 separate jobs.
                assert len(job_ids) == 2
        finally:
            chcall(f"echo y | sacctmgr modify {command} set MaxSubmitJobs=-1")
            reset_max_submit_jobs = executor.get_max_submit_jobs()
            assert reset_max_submit_jobs == original_max_submit_jobs


def test_slurm_max_submit_user_env() -> None:
    max_submit_jobs = 4

    executor = cluster_tools.get_executor("slurm", debug=True)
    original_max_submit_jobs = executor.get_max_submit_jobs()

    os.environ["SLURM_MAX_SUBMIT_JOBS"] = str(max_submit_jobs)
    new_max_submit_jobs = executor.get_max_submit_jobs()

    try:
        assert new_max_submit_jobs == max_submit_jobs

        with executor:
            futures = executor.map_to_futures(square, range(10))

            result = [fut.result() for fut in futures]
            assert result == [i**2 for i in range(10)]

            job_ids = {fut.cluster_jobid for fut in futures}  # type: ignore[attr-defined]
            # The 10 work packages should have been scheduled as 3 separate jobs.
            assert len(job_ids) == 3
    finally:
        del os.environ["SLURM_MAX_SUBMIT_JOBS"]
        reset_max_submit_jobs = executor.get_max_submit_jobs()
        assert reset_max_submit_jobs == original_max_submit_jobs


def test_slurm_deferred_submit() -> None:
    max_submit_jobs = 1

    # Only one job can be scheduled at a time
    call(f"echo y | sacctmgr modify qos normal set MaxSubmitJobs={max_submit_jobs}")
    executor = cluster_tools.get_executor("slurm", debug=True)

    try:
        with executor:
            time_of_start = time.time()
            futures = executor.map_to_futures(sleep, [0.5, 0.5])
            time_of_futures = time.time()
            concurrent.futures.wait(futures)
            time_of_result = time.time()

            # The futures should be returned before each job was scheduled
            assert time_of_futures - time_of_start < 0.5

            # Computing the results should have taken at least two seconds
            # since only one job is scheduled at a time and each job takes 0.5 seconds
            assert time_of_result - time_of_start > 1
    finally:
        call("echo y | sacctmgr modify qos normal set MaxSubmitJobs=-1")


def wait_until_first_job_was_submitted(
    executor: cluster_tools.SlurmExecutor, state: str | None = None
) -> None:
    # Since the job submission is not synchronous, we need to poll
    # to find out when the first job was submitted
    while executor.get_number_of_submitted_jobs(state) <= 0:
        time.sleep(0.1)


@pytest.mark.timeout(30)
def test_slurm_deferred_submit_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that the SlurmExecutor stops scheduling jobs in a separate thread
    # once it was killed even if the executor was used multiple times and
    # therefore started multiple job submission threads
    monkeypatch.setenv("SIGTERM_WAIT_IN_S", "0")
    max_submit_jobs = 1

    # Only one job can be scheduled at a time
    call(f"echo y | sacctmgr modify qos normal set MaxSubmitJobs={max_submit_jobs}")
    executor = cluster_tools.get_executor("slurm", debug=True)

    try:
        with executor:
            # Use the executor twice to start multiple job submission threads
            executor.map_to_futures(sleep, [0.5] * 10)
            executor.map_to_futures(sleep, [0.5] * 10)

            wait_until_first_job_was_submitted(executor)

            for submit_thread in executor.submit_threads:
                assert submit_thread.is_alive()

            sigint_handler = signal.getsignal(signal.SIGINT)
            assert callable(sigint_handler)  # Mainly for typechecking
            sigint_handler(signal.SIGINT, None)

            # Wait for the threads to die down, but less than it would take to submit all jobs
            # which would take ~5 seconds since only one job is scheduled at a time
            for submit_thread in executor.submit_threads:
                submit_thread.join(1)
                assert not submit_thread.is_alive()

            # Wait for scheduled jobs to be canceled, so that the queue is empty again
            while executor.get_number_of_submitted_jobs() > 0:
                time.sleep(0.5)

    finally:
        call("echo y | sacctmgr modify qos normal set MaxSubmitJobs=-1")


@pytest.mark.timeout(30)
def test_slurm_job_canceling_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test that scheduled jobs are canceled on shutdown, regardless
    # of whether they are pending or running.
    monkeypatch.setenv("SLURM_MAX_RUNNING_SIZE", "2")
    monkeypatch.setenv("SIGTERM_WAIT_IN_S", "0")

    executor = cluster_tools.get_executor("slurm", debug=True)

    with executor:
        # Only two jobs can run at once, so that some of the jobs will be
        # running and some will be pending.
        executor.map_to_futures(sleep, [10] * 4)

        # Wait until first job is running
        wait_until_first_job_was_submitted(executor, "RUNNING")

        job_start_time = time.time()

        sigint_handler = signal.getsignal(signal.SIGINT)
        assert callable(sigint_handler)  # Mainly for typechecking
        sigint_handler(signal.SIGINT, None)

        # Wait for scheduled jobs to be canceled, so that the queue is empty again
        # and measure how long the cancellation takes
        while executor.get_number_of_submitted_jobs() > 0:
            time.sleep(0.5)

    job_cancellation_duration = time.time() - job_start_time

    # Killing the executor should have canceled all submitted jobs, regardless
    # of whether they were running or pending in much less time than it would
    # have taken the jobs to finish on their own
    assert job_cancellation_duration < 5


@pytest.mark.timeout(30)
def test_slurm_signal_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIGTERM_WAIT_IN_S", "0")
    original_sigint_handler_was_called = False

    def original_sigint_handler(_signum: int | None, _frame: Any) -> None:
        nonlocal original_sigint_handler_was_called
        original_sigint_handler_was_called = True

    signal.signal(
        signal.SIGINT,
        original_sigint_handler,
    )

    assert len(cluster_tools.SlurmExecutor._shutdown_hooks) == 0
    # Reset class variable after changing signal handler so it will be set again when the next cluster executor is instantiated
    cluster_tools.SlurmExecutor._installed_signal_handler = False

    with cluster_tools.get_executor("slurm", debug=True) as executor1:
        assert executor1.get_number_of_submitted_jobs() == 0
        assert len(cluster_tools.SlurmExecutor._shutdown_hooks) == 1

        futures = executor1.map_to_futures(square, [2])
        concurrent.futures.wait(futures)

    # Let the first executor be no longer referenced to provoke potential bugs in the signal handler chaining
    # See https://github.com/scalableminds/webknossos-libs/pull/1317
    del executor1
    gc.collect()

    with cluster_tools.get_executor("slurm", debug=True) as executor2:
        assert executor2.get_number_of_submitted_jobs() == 0
        assert len(cluster_tools.SlurmExecutor._shutdown_hooks) == 1

        executor2.map_to_futures(sleep, [10] * 4)

        # Wait until first job is running
        wait_until_first_job_was_submitted(executor2, "RUNNING")

        job_start_time = time.time()

        sigint_handler = signal.getsignal(signal.SIGINT)
        assert callable(sigint_handler)  # Mainly for typechecking
        sigint_handler(signal.SIGINT, None)

        assert original_sigint_handler_was_called

        # Wait for scheduled jobs to be canceled, so that the queue is empty again
        # and measure how long the cancellation takes
        while executor2.get_number_of_submitted_jobs() > 0:
            time.sleep(0.5)

        job_cancellation_duration = time.time() - job_start_time

        # Killing the executor should have canceled all submitted jobs, regardless
        # of whether they were running or pending in much less time than it would
        # have taken the jobs to finish on their own
        assert job_cancellation_duration < 5

    assert len(cluster_tools.SlurmExecutor._shutdown_hooks) == 0, (
        "The cluster executor shutdown hooks should have been cleaned up"
    )


def test_slurm_number_of_submitted_jobs() -> None:
    number_of_jobs = 6
    executor = cluster_tools.get_executor("slurm", debug=True)

    assert executor.get_number_of_submitted_jobs() == 0

    with executor:
        futures = executor.map_to_futures(sleep, [1] * number_of_jobs)

        wait_until_first_job_was_submitted(executor)

        assert executor.get_number_of_submitted_jobs() == number_of_jobs

        concurrent.futures.wait(futures)
        time.sleep(3)
        assert executor.get_number_of_submitted_jobs() == 0


@pytest.mark.requires_modified_slurm_config
def test_slurm_max_array_size() -> None:
    expected_max_array_size = 2

    executor = cluster_tools.get_executor("slurm", debug=True)

    max_array_size = executor.get_max_array_size()
    assert max_array_size == expected_max_array_size

    with executor:
        futures = executor.map_to_futures(square, range(6))
        concurrent.futures.wait(futures)
        job_ids = [fut.cluster_jobid for fut in futures]  # type: ignore[attr-defined]

        # Count how often each job_id occurs which corresponds to the array size of the job
        occurrences = list(Counter(job_ids).values())

        assert all(array_size <= expected_max_array_size for array_size in occurrences)


@pytest.mark.skip(
    reason="This test takes more than a minute and is disabled by default. Execute it when modifying the RemoteTimeLimitException code."
)
def test_slurm_time_limit() -> None:
    # Time limit resolution is 1 minute, so request 1 minute
    executor = cluster_tools.get_executor(
        "slurm", debug=True, job_resources={"time": "0-00:01:00"}
    )

    with executor:
        # Schedule a job that runs for more than 1 minute
        futures = executor.map_to_futures(sleep, [80])
        concurrent.futures.wait(futures)

        # Job should have been killed with a RemoteTimeLimitException
        assert all(
            isinstance(fut.exception(), cluster_tools.RemoteTimeLimitException)
            for fut in futures
        )


@pytest.mark.requires_modified_slurm_config
def test_slurm_memory_limit() -> None:
    # Request 30 MB
    executor = cluster_tools.get_executor(
        "slurm",
        debug=True,
        job_resources={"mem": "30M"},  # 30M is the smallest limit enforced by Cgroups
    )

    with executor:
        # Schedule a job that allocates more than 30 MB and let it run for more than 1 second
        # because the frequency of the memory polling is 1 second
        duration = 3
        futures = executor.map_to_futures(
            partial(allocate, duration), [1024 * 1024 * 50]
        )
        concurrent.futures.wait(futures)

        # Job should have been killed with a RemoteOutOfMemoryException
        assert all(
            isinstance(fut.exception(), cluster_tools.RemoteOutOfMemoryException)
            for fut in futures
        )


def test_slurm_max_array_size_env() -> None:
    max_array_size = 2

    executor = cluster_tools.get_executor("slurm", debug=True)
    original_max_array_size = executor.get_max_array_size()

    os.environ["SLURM_MAX_ARRAY_SIZE"] = str(max_array_size)
    new_max_array_size = executor.get_max_array_size()

    try:
        assert new_max_array_size == max_array_size

        with executor:
            futures = executor.map_to_futures(square, range(6))
            concurrent.futures.wait(futures)
            job_ids = [fut.cluster_jobid for fut in futures]  # type: ignore[attr-defined]

            # Count how often each job_id occurs which corresponds to the array size of the job
            occurrences = list(Counter(job_ids).values())

            assert all(array_size <= max_array_size for array_size in occurrences)
    finally:
        del os.environ["SLURM_MAX_ARRAY_SIZE"]
        reset_max_array_size = executor.get_max_array_size()
        assert reset_max_array_size == original_max_array_size


test_output_str = "Test-Output"


def log(string: str) -> None:
    logging.debug(string)


def test_pickled_logging() -> None:
    def execute_with_log_level(log_level: int) -> str:
        logging_config = {"level": log_level}
        with cluster_tools.get_executor(
            "slurm",
            debug=True,
            job_resources={"mem": "10M"},
            logging_config=logging_config,
        ) as executor:
            fut = executor.submit(log, test_output_str)
            fut.result()

            output = f".cfut/slurmpy.{fut.cluster_jobid}.log.stdout"  # type: ignore[attr-defined]

            with open(output) as file:
                return file.read()

    debug_out = execute_with_log_level(logging.DEBUG)
    assert test_output_str in debug_out

    info_out = execute_with_log_level(logging.INFO)
    assert test_output_str not in info_out


def test_tailed_logging() -> None:
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


def fail(_val: Any) -> None:
    raise Exception("Fail()")


def output_pickle_path_getter(tmp_dir: str, chunk: int) -> Path:
    return Path(tmp_dir) / f"test_{chunk}.pickle"


def test_preliminary_file_submit() -> None:
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
            assert preliminary_output_path.exists(), (
                "Preliminary output file should exist"
            )
            assert not output_pickle_path.exists(), "Final output file should not exist"

            # Schedule succeeding job with same output path
            fut_2 = executor.submit(
                square,
                3,
                __cfut_options={"output_pickle_path": str(output_pickle_path)},  # type: ignore[call-arg]
            )
            assert fut_2.result() == 9
            assert output_pickle_path.exists(), "Final output file should exist"
            assert not preliminary_output_path.exists(), (
                "Preliminary output file should not exist anymore"
            )


def test_executor_args() -> None:
    def pass_with(exc: cluster_tools.SlurmExecutor) -> None:
        with exc:
            pass

    pass_with(
        cluster_tools.get_executor(
            "slurm", job_resources={"mem": "10M"}, non_existent_arg=True
        )
    )
    # Test should succeed if the above lines don't raise an exception


def test_preliminary_file_map() -> None:
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
            for fut in futs:
                with pytest.raises(Exception):
                    fut.result()

            for idx in a_range:
                output_pickle_path = Path(output_pickle_path_getter(tmp_dir, idx))
                preliminary_output_path = Path(f"{output_pickle_path}.preliminary")

                assert preliminary_output_path.exists(), (
                    "Preliminary output file should exist"
                )
                assert not output_pickle_path.exists(), (
                    "Final output file should not exist"
                )

            # Schedule succeeding jobs with same output paths
            futs_2 = executor.map_to_futures(
                square,
                list(a_range),
                output_pickle_path_getter=partial(output_pickle_path_getter, tmp_dir),
            )
            for fut_2, job_index in zip(futs_2, a_range):
                assert fut_2.result() == square(job_index)

            for idx in a_range:
                output_pickle_path = Path(output_pickle_path_getter(tmp_dir, idx))
                preliminary_output_path = Path(f"{output_pickle_path}.preliminary")
                assert output_pickle_path.exists(), "Final output file should exist"
                assert not preliminary_output_path.exists(), (
                    "Preliminary output file should not exist anymore"
                )


def test_cpu_bind_regression() -> None:
    os.environ["SLURM_CPU_BIND"] = (
        "quiet,mask_cpu:0x000000000000040000000000000000040000"
    )

    stdout, _ = chcall("scontrol show config | sed -n '/^TaskPlugin/s/.*= *//p'")
    assert "task/affinity" in stdout, (
        "The task/affinity TaskPlugin needs to be enabled in order for SLURM_CPU_BIND to have an effect."
    )

    with cluster_tools.get_executor("slurm") as executor:
        # The slurm job should not fail, although an invalid CPU mask was set before the submission
        # See https://bugs.schedmd.com/show_bug.cgi?id=14298
        future = executor.submit(square, 2)
        assert future.result() == 4

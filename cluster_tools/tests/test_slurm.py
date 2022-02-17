import concurrent.futures
import contextlib
import io
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from collections import Counter
from functools import partial
from pathlib import Path

import pytest

import cluster_tools
from cluster_tools.util import call


# "Worker" functions.
def square(n):
    return n * n


def sleep(duration):
    time.sleep(duration)
    return duration


logging.basicConfig()


def expect_fork():
    assert mp.get_start_method() == "fork"
    return True


def test_map_with_spawn():
    with cluster_tools.get_executor(
        "slurm", max_workers=5, start_method="spawn"
    ) as executor:
        assert executor.submit(
            expect_fork
        ).result(), "Slurm should ignore provided start_method"


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


def test_slurm_max_submit_user():
    max_submit_jobs = 6

    # MaxSubmitJobs can either be defined at the user or at the qos level
    for command in ["user root", "qos normal"]:
        executor = cluster_tools.get_executor("slurm", debug=True)
        original_max_submit_jobs = executor.get_max_submit_jobs()

        _, _, exit_code = call(
            f"echo y | sacctmgr modify {command} set MaxSubmitJobs={max_submit_jobs}"
        )
        try:
            assert exit_code == 0

            new_max_submit_jobs = executor.get_max_submit_jobs()
            assert new_max_submit_jobs == max_submit_jobs

            with executor:
                futures = executor.map_to_futures(square, range(10))

                result = [fut.result() for fut in futures]
                assert result == [i ** 2 for i in range(10)]

                job_ids = {fut.cluster_jobid for fut in futures}
                # The 10 work packages should have been scheduled as 2 separate jobs.
                assert len(job_ids) == 2
        finally:
            _, _, exit_code = call(
                f"echo y | sacctmgr modify {command} set MaxSubmitJobs=-1"
            )
            assert exit_code == 0
            reset_max_submit_jobs = executor.get_max_submit_jobs()
            assert reset_max_submit_jobs == original_max_submit_jobs


def test_slurm_max_submit_user_env():
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
            assert result == [i ** 2 for i in range(10)]

            job_ids = {fut.cluster_jobid for fut in futures}
            # The 10 work packages should have been scheduled as 3 separate jobs.
            assert len(job_ids) == 3
    finally:
        del os.environ["SLURM_MAX_SUBMIT_JOBS"]
        reset_max_submit_jobs = executor.get_max_submit_jobs()
        assert reset_max_submit_jobs == original_max_submit_jobs


def test_slurm_deferred_submit():
    max_submit_jobs = 1

    # Only one job can be scheduled at a time
    _, _, exit_code = call(
        f"echo y | sacctmgr modify qos normal set MaxSubmitJobs={max_submit_jobs}"
    )
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
        _, _, exit_code = call(
            "echo y | sacctmgr modify qos normal set MaxSubmitJobs=-1"
        )


def wait_until_first_job_was_submitted(executor):
    # Since the job submission is not synchronous, we need to poll
    # to find out when the first job was submitted
    while executor.get_number_of_submitted_jobs() <= 0:
        time.sleep(0.1)


def test_slurm_deferred_submit_shutdown():
    # Test that the SlurmExecutor stops scheduling jobs in a separate thread
    # once it was killed even if the executor was used multiple times and
    # therefore started multiple job submission threads
    max_submit_jobs = 1

    # Only one job can be scheduled at a time
    _, _, exit_code = call(
        f"echo y | sacctmgr modify qos normal set MaxSubmitJobs={max_submit_jobs}"
    )
    executor = cluster_tools.get_executor("slurm", debug=True)

    try:
        # Use the executor twice to start multiple job submission threads
        executor.map_to_futures(sleep, [0.5] * 10)
        executor.map_to_futures(sleep, [0.5] * 10)

        wait_until_first_job_was_submitted(executor)

        for submit_thread in executor.submit_threads:
            assert submit_thread.is_alive()

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            executor.handle_kill(None, None)
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 130

        # Wait for the threads to die down, but less than it would take to submit all jobs
        # which would take ~5 seconds since only one job is scheduled at a time
        for submit_thread in executor.submit_threads:
            submit_thread.join(1)
            assert not submit_thread.is_alive()

        # Wait for scheduled jobs to finish, so that the queue is empty again
        while executor.get_number_of_submitted_jobs() > 0:
            time.sleep(0.5)

    finally:
        _, _, exit_code = call(
            "echo y | sacctmgr modify qos normal set MaxSubmitJobs=-1"
        )


def test_slurm_number_of_submitted_jobs():
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


def test_slurm_max_array_size():
    max_array_size = 2

    executor = cluster_tools.get_executor("slurm", debug=True)
    original_max_array_size = executor.get_max_array_size()

    command = f"MaxArraySize={max_array_size}"
    _, _, exit_code = call(
        f"echo -e '{command}' >> /etc/slurm/slurm.conf && scontrol reconfigure"
    )

    try:
        assert exit_code == 0

        new_max_array_size = executor.get_max_array_size()
        assert new_max_array_size == max_array_size

        with executor:
            futures = executor.map_to_futures(square, range(6))
            concurrent.futures.wait(futures)
            job_ids = [fut.cluster_jobid for fut in futures]

            # Count how often each job_id occurs which corresponds to the array size of the job
            occurences = list(Counter(job_ids).values())

            assert all(array_size <= max_array_size for array_size in occurences)
    finally:
        _, _, exit_code = call(
            f"sed -i 's/{command}//g' /etc/slurm/slurm.conf && scontrol reconfigure"
        )
        assert exit_code == 0
        reset_max_array_size = executor.get_max_array_size()
        assert reset_max_array_size == original_max_array_size


def test_slurm_max_array_size_env():
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
            job_ids = [fut.cluster_jobid for fut in futures]

            # Count how often each job_id occurs which corresponds to the array size of the job
            occurences = list(Counter(job_ids).values())

            assert all(array_size <= max_array_size for array_size in occurences)
    finally:
        del os.environ["SLURM_MAX_ARRAY_SIZE"]
        reset_max_array_size = executor.get_max_array_size()
        assert reset_max_array_size == original_max_array_size


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


def fail(val):
    raise Exception("Fail()")


def output_pickle_path_getter(tmp_dir, chunk):

    return Path(tmp_dir) / f"test_{chunk}.pickle"


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


def test_executor_args():
    def pass_with(exc):
        with exc:
            pass

    pass_with(
        cluster_tools.get_executor(
            "slurm", job_resources={"mem": "10M"}, non_existent_arg=True
        )
    )
    # Test should succeed if the above lines don't raise an exception


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
            for fut in futs:
                with pytest.raises(Exception):
                    fut.result()

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
                assert fut.result() == square(job_index)

            for idx in a_range:
                output_pickle_path = Path(output_pickle_path_getter(tmp_dir, idx))
                preliminary_output_path = Path(f"{output_pickle_path}.preliminary")
                assert output_pickle_path.exists(), "Final output file should exist"
                assert (
                    not preliminary_output_path.exists()
                ), "Preliminary output file should not exist anymore"

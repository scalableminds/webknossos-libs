import cluster_tools
import subprocess
import concurrent.futures
import time
import sys

# "Worker" functions.
def square(n):
    return n * n


def sleep(duration):
    time.sleep(duration)
    return duration


def get_executors():
    return [
        cluster_tools.get_executor(
            "slurm", debug=True, keep_logs=True, job_resources={"mem": "10M"}
        ),
        cluster_tools.get_executor("multiprocessing", 5),
        cluster_tools.get_executor("sequential"),
    ]


def test_submit():
    def run_square_numbers(executor):
        with executor:
            job_count = 5
            job_range = range(job_count)
            futures = [executor.submit(square, n) for n in job_range]
            for future, job_index in zip(futures, job_range):
                assert future.result() == square(job_index)

    for exc in get_executors():
        run_square_numbers(exc)


def test_unordered_sleep():
    """Get host identifying information about the servers running
    our jobs.
    """

    def run_sleeps(executor):
        with executor:
            durations = [10, 5, 15]
            futures = [executor.submit(sleep, n) for n in durations]
            if not isinstance(executor, cluster_tools.SequentialExecutor):
                durations.sort()
            for duration, future in zip(
                durations, concurrent.futures.as_completed(futures)
            ):
                assert future.result() == duration

    for exc in get_executors():
        run_sleeps(exc)


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


def test_slurm_submit_returns_job_ids():
    exc = cluster_tools.get_executor("slurm", debug=True, keep_logs=True)
    with exc:
        future = exc.submit(square, 2)
        assert isinstance(future.slurm_jobid, int)
        assert future.slurm_jobid > 0
        assert future.result() == 4

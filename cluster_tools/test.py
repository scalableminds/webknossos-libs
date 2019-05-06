import cluster_tools
import subprocess
import concurrent.futures
import time
import sys
import logging

# "Worker" functions.
def square(n):
    return n * n


def sleep(duration):
    time.sleep(duration)
    return duration

logging.basicConfig()


def get_executors():
    return [
        cluster_tools.get_executor(
            "slurm", debug=True, keep_logs=True, job_resources={"mem": "10M"}
        ),
        cluster_tools.get_executor("multiprocessing", max_workers=5),
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


def test_executor_args():
    def pass_with(exc):
        with exc:
            pass

    pass_with(cluster_tools.get_executor(
        "slurm", job_resources={"mem": "10M"}, non_existent_arg=True
    ))
    pass_with(cluster_tools.get_executor("multiprocessing", non_existent_arg=True))
    pass_with(cluster_tools.get_executor("sequential", non_existent_arg=True))

    # Test should succeed if the above lines don't raise an exception


test_output_str = "Test-Output"
def log():
    logging.debug(test_output_str)

def test_pickled_logging():

    def execute_with_log_level(log_level):
        logging_config = {
            "level": log_level,
        }
        with cluster_tools.get_executor(
            "slurm", debug=True, keep_logs=True, job_resources={"mem": "10M"}, logging_config=logging_config
        ) as executor:
            fut = executor.submit(log)
            fut.result()

            output = ".cfut/slurmpy.stdout.{}.log".format(fut.slurm_jobid)

            with open(output, 'r') as file:
                return file.read()

    debug_out = execute_with_log_level(logging.DEBUG)
    assert test_output_str in debug_out

    debug_out = execute_with_log_level(logging.INFO)
    assert not (test_output_str in debug_out)
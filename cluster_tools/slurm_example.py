import cluster_tools
import subprocess
import concurrent.futures
import time

# "Worker" functions.
def square(n):
    return n * n

def hostinfo():
    return subprocess.check_output('uname -a', shell=True)

def sleep(duration):
    time.sleep(duration)
    return True


def example_1():
    """Square some numbers on remote hosts!
    """
    executor = cluster_tools.get_executor("slurm", debug=True, keep_logs=True)
    # executor = cluster_tools.get_executor("multiprocessing", 5)
    # executor = cluster_tools.get_executor("sequential")
    with executor:
        job_count = 5
        futures = [executor.submit(square, n) for n in range(job_count)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def example_2():
    """Get host identifying information about the servers running
    our jobs.
    """
    with cluster_tools.SlurmExecutor(False) as executor:
        futures = [executor.submit(hostinfo) for n in range(15)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result().strip())


def example_3():
    with cluster_tools.SlurmExecutor(False) as exc:
        print(list(exc.map(square, [5, 7, 11, 12, 13, 14, 15, 16, 17])))


def sleep_example():
    executor = cluster_tools.get_executor("slurm")
    # executor = cluster_tools.get_executor("multiprocessing", 5)
    # executor = cluster_tools.get_executor("sequential")
    with executor:
        print(list(executor.map(sleep, [10, 10, 10])))


if __name__ == '__main__':
    example_1()
    example_2()
    example_3()
    sleep_example()

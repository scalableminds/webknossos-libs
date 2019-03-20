import cluster_tools
import subprocess
import concurrent.futures

# "Worker" functions.
def square(n):
    return n * n
def hostinfo():
    return subprocess.check_output('uname -a', shell=True)

def example_1():
    """Square some numbers on remote hosts!
    """
    with cluster_tools.SlurmExecutor(True, keep_logs=True) as executor:
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
    """Demonstrates the use of the map() convenience function.
    """
    exc = cluster_tools.SlurmExecutor(False)
    print(list(cluster_tools.map(exc, square, [5, 7, 11])))

if __name__ == '__main__':
    example_1()
    example_2()
    example_3()

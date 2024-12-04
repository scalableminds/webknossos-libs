import logging
import multiprocessing as mp

import cluster_tools

logging.basicConfig()


def expect_fork() -> bool:
    assert mp.get_start_method() == "fork"
    return True


def expect_forkserver() -> bool:
    assert mp.get_start_method() == "forkserver"
    return True


def expect_spawn() -> bool:
    assert mp.get_start_method() == "spawn"
    return True


def test_map_with_spawn() -> None:
    with cluster_tools.get_executor("multiprocessing", max_workers=5) as executor:
        assert executor.submit(
            expect_spawn
        ).result(), "Multiprocessing should use `spawn` by default"

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method=None
    ) as executor:
        assert executor.submit(
            expect_spawn
        ).result(), "Multiprocessing should use `spawn` if start_method is None"

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method="forkserver"
    ) as executor:
        assert executor.submit(
            expect_forkserver
        ).result(), "Multiprocessing should use `forkserver` if requested"

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method="fork"
    ) as executor:
        assert executor.submit(
            expect_fork
        ).result(), "Multiprocessing should use `fork` if requested"


def test_executor_args() -> None:
    def pass_with(exc: cluster_tools.MultiprocessingExecutor) -> None:
        with exc:
            pass

    pass_with(cluster_tools.get_executor("multiprocessing", non_existent_arg=True))
    # Test should succeed if the above lines don't raise an exception


def test_multiprocessing_validation() -> None:
    import sys
    from subprocess import PIPE, STDOUT, Popen

    cmd = [sys.executable, "guardless_multiprocessing.py"]
    p = Popen(cmd, shell=False, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    assert p.stdout is not None
    output = p.stdout.read()

    assert "current process has finished its bootstrapping phase." in str(output), "S"

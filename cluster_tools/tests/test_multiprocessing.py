import logging
import multiprocessing as mp
import os

import pytest

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


def accept_high_mem(data: str) -> int:
    return len(data)


@pytest.mark.skip(
    reason="This test does not pass on the CI. Probably because the machine does not have enough RAM."
)
def test_high_ram_usage() -> None:
    very_long_string = " " * 10**6 * 2500

    os.environ["MULTIPROCESSING_VIA_IO"] = "True"

    with cluster_tools.get_executor("multiprocessing") as executor:
        fut1 = executor.submit(
            accept_high_mem,
            very_long_string,
            __cfut_options={"output_pickle_path": "/tmp/test.pickle"},  # type: ignore[call-arg]
        )
        assert fut1.result() == len(very_long_string)

        os.environ["MULTIPROCESSING_VIA_IO_TMP_DIR"] = "."
        fut2 = executor.submit(accept_high_mem, very_long_string)
        assert fut2.result() == len(very_long_string)

    del os.environ["MULTIPROCESSING_VIA_IO"]


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

import concurrent
import logging
import multiprocessing as mp
import tempfile
from pathlib import Path

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
        assert executor.submit(expect_spawn).result(), (
            "Multiprocessing should use `spawn` by default"
        )

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method=None
    ) as executor:
        assert executor.submit(expect_spawn).result(), (
            "Multiprocessing should use `spawn` if start_method is None"
        )

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method="forkserver"
    ) as executor:
        assert executor.submit(expect_forkserver).result(), (
            "Multiprocessing should use `forkserver` if requested"
        )

    with cluster_tools.get_executor(
        "multiprocessing", max_workers=5, start_method="fork"
    ) as executor:
        assert executor.submit(expect_fork).result(), (
            "Multiprocessing should use `fork` if requested"
        )


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


def log_debug(string: str) -> None:
    logging.debug(string)


test_output_strs = ["Test-Output-1", "Test-Output-2"]


@pytest.mark.parametrize(
    "start_method",
    ["forkserver", "spawn"],
)
def test_logging(start_method: str) -> None:
    def execute_with_log_level(log_level: int) -> str:
        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            file_handler = None
            root_logger = None
            try:
                log_file_path = Path(tmp_dir) / "test-log-file.log"
                root_logger = logging.getLogger()
                file_handler = logging.FileHandler(
                    log_file_path, mode="w", encoding="UTF-8"
                )
                file_handler.setLevel(log_level)
                root_logger.addHandler(file_handler)
                log_debug(test_output_strs[0])
                log_debug(test_output_strs[1])
                with cluster_tools.get_executor(
                    "multiprocessing",
                    max_workers=2,
                    start_method=start_method,
                ) as executor:
                    futures = executor.map_to_futures(log_debug, test_output_strs)
                    concurrent.futures.wait(futures)

                with open(log_file_path) as file:
                    return file.read()

            finally:
                if root_logger and file_handler:
                    root_logger.removeHandler(file_handler)

    debug_out = execute_with_log_level(logging.DEBUG)
    print(f"Debug out {debug_out}")
    assert all(string in debug_out for string in test_output_strs)

    info_out = execute_with_log_level(logging.INFO)
    print(f"Info out {info_out}")
    assert all(string not in info_out for string in test_output_strs)

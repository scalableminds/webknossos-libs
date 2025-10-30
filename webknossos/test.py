# /// script
# dependencies = [
#   "requests",
# ]
# requires-python = ">=3.10"
# ///

# ruff: noqa: T201

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj, rmtree, unpack_archive
from time import sleep
from typing import Literal

import requests

WK_TOKEN = "1b88db86331a38c21a0b235794b9e459856490d70408bcffb767f64ade0f83d2bdb4c4e181b9a9a30cdece7cb7c65208cc43b6c1bb5987f5ece00d348b1a905502a266f8fc64f0371cd6559393d72e031d0c2d0cabad58cccf957bb258bc86f05b5dc3d4fff3d5e3d9c0389a6027d861a21e78e3222fb6c5b7944520ef21761e"
WK_URL = "http://localhost:9000"
WK_API_VERSION = 12
IS_WINDOWS = sys.platform == "win32"
PROXAY_VERSION = "1.9.0"


def check_and_clean_datasets_folder() -> None:
    # Within the tests folder is a binaryData folder of the local running webknossos instance. This folder is cleaned up before running the tests.
    # This find command gets all directories in binaryData/Organization_X except for the l4_sample and e2006_knossos directories and deletes them.
    for dataset_path in Path("tests/binaryData/Organization_X").iterdir():
        if dataset_path.is_dir() and not (
            dataset_path.name == "l4_sample" or dataset_path.name == "e2006_knossos"
        ):
            rmtree(dataset_path)


def download_and_unpack(
    url: str | None,
    org_binary_data_dir: Path,
) -> None:
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        with open(org_binary_data_dir / "tmp.zip", "wb") as f:
            copyfileobj(req.raw, f)
    unpack_archive(org_binary_data_dir / "tmp.zip", org_binary_data_dir)
    (org_binary_data_dir / "tmp.zip").unlink()


def start_wk_via_docker(wk_docker_dir: Path, wk_docker_tag: str) -> None:
    print(
        f"Starting webknossos via docker compose with tag {wk_docker_tag}",
        flush=True,
    )

    subprocess.check_call(
        ["docker", "compose", "pull", "webknossos"], cwd=wk_docker_dir
    )

    # Create the binaryData directory and download the l4_sample dataset
    binary_data_dir = wk_docker_dir / "binaryData"
    org_binary_data_dir = binary_data_dir / "Organization_X"
    if not (org_binary_data_dir / "l4_sample").exists():
        org_binary_data_dir.mkdir(parents=True, exist_ok=True)
        download_and_unpack(
            "https://static.webknossos.org/data/l4_sample.zip", org_binary_data_dir
        )

    # Start the webknossos server
    subprocess.check_call(
        ["docker", "compose", "up", "-d", "--no-build", "webknossos"],
        cwd=wk_docker_dir,
        env={
            **os.environ,
            "USER_UID": str(os.getuid()),
            "USER_GID": str(os.getgid()),
        },
    )

    # Wait for booting
    while not is_wk_healthy():
        sleep(1)

    # Prepare the test database
    subprocess.check_call(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "webknossos",
            "tools/postgres/dbtool.js",
            "prepare-test-db",
        ],
        cwd=wk_docker_dir,
    )


def is_wk_healthy():
    try:
        if requests.get(f"{WK_URL}/api/v{WK_API_VERSION}/health").ok:
            return True
    except requests.RequestException:
        pass
    return False


@contextmanager
def local_test_wk() -> Iterator[None]:
    assert not IS_WINDOWS, "Windows is not supported for local testing"

    check_and_clean_datasets_folder()

    # Fetch current version of webknossos.org this can be replaced with a fixed version for testing
    wk_version = requests.get(
        f"https://webknossos.org/api/v{WK_API_VERSION}/buildinfo"
    ).json()["webknossos"]["version"]
    wk_docker_tag = f"master__{wk_version}"
    os.environ["DOCKER_TAG"] = wk_docker_tag
    wk_docker_dir = Path("tests")
    tear_down_wk = False

    try:
        if not is_wk_healthy():
            start_wk_via_docker(wk_docker_dir, wk_docker_tag)
            tear_down_wk = True
        else:
            print(
                f"Using the already running webknossos at {WK_URL}. Make sure l4_sample exists and is set to public first!",
                flush=True,
            )

        # Check that the login user is set up correctly
        user_req = requests.get(
            f"{WK_URL}/api/v{WK_API_VERSION}/user", headers={"X-Auth-Token": WK_TOKEN}
        )
        if not user_req.ok or "user_a@scalableminds.com" not in user_req.text:
            print(
                """The login user user_a@scalableminds.com could not be found or changed.
Please ensure that the test-db is prepared by running this in the webknossos repo
(⚠️ this overwrites your local webknossos database):
"tools/postgres/dbtool.js prepare-test-db""",
                flush=True,
            )
            raise RuntimeError("Login user could not be found or changed.")

        # Trigger dataset import via directory scan
        requests.post(
            f"{WK_URL}/data/v{WK_API_VERSION}/triggers/checkInboxBlocking",
            headers={"X-Auth-Token": WK_TOKEN},
        )
        yield
    finally:
        if tear_down_wk:
            subprocess.check_call(["docker", "compose", "down"], cwd=wk_docker_dir)


@contextmanager
def proxay(mode: Literal["record", "replay"], quiet: bool) -> Iterator[None]:
    # Ensure that proxay is installed, otherwise it will lead to hard-to-debug errors
    try:
        subprocess.check_output(
            ["npx", "-y", f"proxay@{PROXAY_VERSION}"],
            text=True,
            stderr=subprocess.STDOUT,
            shell=IS_WINDOWS,
        )
    except subprocess.CalledProcessError as e:
        # Checking that proxay is installed properly
        if "Please specify a valid mode (record or replay)" not in e.output:
            raise

    cmd = ["npx", f"proxay@{PROXAY_VERSION}"]
    if mode == "record":
        cmd += [
            "--mode",
            "record",
            "--host",
            WK_URL,
            "--tapes-dir",
            "tests/cassettes",
        ]
    elif mode == "replay":
        cmd += ["--mode", "replay", "--tapes-dir", "tests/cassettes"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    proxay_process = None
    try:
        print(f"Starting proxay with command: {cmd} {quiet=}", flush=True)
        if quiet:
            proxay_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=IS_WINDOWS,
            )
        else:
            proxay_process = subprocess.Popen(
                cmd,
                shell=IS_WINDOWS,
            )
        sleep(1)
        yield
    finally:
        if proxay_process is not None:
            print("Terminating proxay", flush=True)
            proxay_process.terminate()
            proxay_process.wait(5)
            proxay_process.kill()


def run_pytest(args: list[str]) -> None:
    process = subprocess.Popen(args)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("Terminating pytest", flush=True)
        if IS_WINDOWS:
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
        process.wait()
        raise


def main(snapshot_command: Literal["refresh", "add"] | None, args: list[str]) -> None:
    python_version = os.environ.get("PYTHON_VERSION", "3.13")

    # Using forkserver instead of spawn is faster. Fork should never be used due to potential deadlock problems.
    os.environ["MULTIPROCESSING_DEFAULT_START_METHOD"] = os.environ.get(
        "MULTIPROCESSING_DEFAULT_START_METHOD", "forkserver"
    )

    # Export the necessary environment variables
    os.environ["WK_TOKEN"] = WK_TOKEN
    os.environ["WK_URL"] = WK_URL

    # Note that pytest should be executed via `python -m`, since
    # this will ensure that the current directory is added to sys.path
    # (which is standard python behavior). This is necessary so that the imports
    # refer to the checked out (and potentially modified) code.
    pytest_cmd = [
        "uv",
        "run",
        "--all-extras",
        "--python",
        python_version,
        "-m",
        "pytest",
        "--suppress-no-test-exit-code",
        "-vv",
    ]

    if snapshot_command == "refresh":
        rmtree("tests/cassettes", ignore_errors=True)

        with proxay("record", quiet=False), local_test_wk():
            run_pytest(pytest_cmd + ["-m", "use_proxay"] + args)
    elif snapshot_command == "add":
        with proxay("record", quiet=False), local_test_wk():
            run_pytest(pytest_cmd + ["-m", "use_proxay"] + args)
    else:
        with proxay("replay", quiet=False):
            run_pytest(pytest_cmd + ["--timeout=360"] + args)


if __name__ == "__main__":
    snapshot_command = None
    args = sys.argv[1:]
    if len(args) > 0 and args[0] in ["--refresh-snapshots", "--add-snapshots"]:
        snapshot_command = args[0][2:-10]
        args = args[1:]

    main(snapshot_command, args)

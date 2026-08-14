import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import sleep

from upath import UPath

from webknossos.utils import rmtree

TESTDATA_DIR = UPath(__file__).parent.parent / "testdata"
TESTOUTPUT_DIR = UPath(__file__).parent.parent / "testoutput"


S3_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
S3_PORT = "8000"
# rustfs binds a web console in addition to the S3 API; pin it to a fixed,
# unlikely-to-collide port instead of its default (:9001), which may already
# be in use by something else on the host.
RUSTFS_CONSOLE_PORT = "8001"

REMOTE_TESTOUTPUT_DIR = UPath(
    "s3://testoutput",
    key=S3_ROOT_USER,
    secret=S3_ROOT_PASSWORD,
    endpoint_url=f"http://localhost:{S3_PORT}",
)

# Cache dir for the auto-downloaded RustFS binary (gitignored).
RUSTFS_BIN_DIR = Path(__file__).parent.parent / ".rustfs_bin"
RUSTFS_RELEASES_API_URL = (
    "https://api.github.com/repos/rustfs/rustfs/releases?per_page=1"
)


def _rustfs_asset_pattern() -> str:
    """Maps the current OS/arch to the RustFS release-asset filename prefix."""
    system = platform.system()
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    if system == "Linux":
        return "rustfs-linux-aarch64-gnu-" if is_arm else "rustfs-linux-x86_64-gnu-"
    if system == "Darwin":
        if is_arm:
            return "rustfs-macos-aarch64-"
        raise RuntimeError(
            "RustFS does not publish a prebuilt binary for Intel macOS "
            "(only Apple Silicon). Either build RustFS from source "
            "(see https://github.com/rustfs/rustfs, "
            "`./build-rustfs.sh --platform x86_64-apple-darwin`) and place "
            f"the resulting `rustfs` binary at {RUSTFS_BIN_DIR / 'rustfs'}, "
            "or install Docker and run "
            "`docker run -p 8000:9000 rustfs/rustfs server /data` manually."
        )
    if system == "Windows":
        return "rustfs-windows-x86_64-"
    raise RuntimeError(f"Unsupported platform for RustFS: {system}")


def _download_rustfs_release(asset_pattern: str) -> Path:
    request = urllib.request.Request(
        RUSTFS_RELEASES_API_URL, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            releases = json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                "Failed to query the GitHub releases API for RustFS "
                "(likely rate-limited). Please wait a bit and retry, or "
                "download a release manually from "
                f"https://github.com/rustfs/rustfs/releases into {RUSTFS_BIN_DIR}."
            ) from e
        raise

    release = releases[0]
    asset = next(
        (
            a
            for a in release["assets"]
            if a["name"].startswith(asset_pattern) and a["name"].endswith(".zip")
        ),
        None,
    )
    if asset is None:
        raise RuntimeError(
            f"Could not find a RustFS release asset matching {asset_pattern!r} "
            f"in release {release['tag_name']}."
        )

    RUSTFS_BIN_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RUSTFS_BIN_DIR / asset["name"]
    urllib.request.urlretrieve(asset["browser_download_url"], archive_path)
    try:
        shutil.unpack_archive(archive_path, RUSTFS_BIN_DIR)
    finally:
        archive_path.unlink()
    return RUSTFS_BIN_DIR


def _ensure_rustfs_binary() -> Path:
    binary_name = "rustfs.exe" if sys.platform == "win32" else "rustfs"
    binary_path = RUSTFS_BIN_DIR / binary_name
    if binary_path.exists():
        return binary_path

    extracted_dir = _download_rustfs_release(_rustfs_asset_pattern())

    # The zip may extract the binary directly, or nested in a subdirectory;
    # find it and flatten it to the expected top-level location.
    found = next(extracted_dir.rglob(binary_name), None)
    assert found is not None, (
        f"Could not find {binary_name!r} after unpacking the RustFS release "
        f"into {extracted_dir}."
    )
    if found != binary_path:
        shutil.move(str(found), str(binary_path))
        for entry in extracted_dir.iterdir():
            if entry != binary_path and entry.is_dir():
                rmtree(UPath(entry))

    if sys.platform != "win32":
        binary_path.chmod(
            binary_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )

    return binary_path


@contextmanager
def use_rustfs() -> Iterator[None]:
    """RustFS is an S3 clone and is used as local test server.

    The matching binary for the current OS/arch is downloaded from GitHub
    releases and cached in RUSTFS_BIN_DIR on first use.

    A no-op on Windows: rustfs's Windows binary hangs under sustained S3
    traffic (looks like an upstream concurrency bug,
    https://github.com/rustfs/rustfs), so all S3-touching tests are marked
    `skip_on_windows` and don't need a running server there.
    """
    if sys.platform == "win32":
        yield
        return

    rustfs_bin = _ensure_rustfs_binary()
    rustfs_path = UPath("testoutput_rustfs")
    rmtree(rustfs_path)
    # Unlike minio, rustfs does not create its volume directory on its own.
    rustfs_path.mkdir(parents=True, exist_ok=True)
    rustfs_process = subprocess.Popen(
        [
            str(rustfs_bin),
            "server",
            "--address",
            f":{S3_PORT}",
            "--console-address",
            f":{RUSTFS_CONSOLE_PORT}",
            str(rustfs_path.absolute()),
        ],
        env={
            **os.environ,
            "RUSTFS_ACCESS_KEY": S3_ROOT_USER,
            "RUSTFS_SECRET_KEY": S3_ROOT_PASSWORD,
        },
    )
    sleep(3)
    assert rustfs_process.poll() is None
    REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
    try:
        yield
    finally:
        rustfs_process.terminate()
        sleep(1)
        rmtree(rustfs_path)

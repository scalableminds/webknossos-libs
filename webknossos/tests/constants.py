import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Iterator

from upath import UPath

from webknossos.utils import rmtree

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTOUTPUT_DIR = Path(__file__).parent.parent / "testoutput"


MINIO_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
MINIO_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
MINIO_PORT = "8000"

REMOTE_TESTOUTPUT_DIR = UPath(
    "s3://testoutput",
    key=MINIO_ROOT_USER,
    secret=MINIO_ROOT_PASSWORD,
    client_kwargs={"endpoint_url": f"http://localhost:{MINIO_PORT}"},
)


@contextmanager
def use_minio() -> Iterator[None]:
    """Minio is an S3 clone and is used as local test server"""
    if sys.platform == "darwin":
        minio_path = Path("testoutput_minio")
        rmtree(minio_path)
        minio_process = subprocess.Popen(
            shlex.split(f"minio server --address :8000 ./{minio_path}"),
            env={
                **os.environ,
                "MINIO_ROOT_USER": MINIO_ROOT_USER,
                "MINIO_ROOT_PASSWORD": MINIO_ROOT_PASSWORD,
            },
        )
        sleep(3)
        assert minio_process.poll() is None
        REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
        try:
            yield
        finally:
            minio_process.terminate()
            rmtree(minio_path)
    else:
        container_name = "minio"
        cmd = (
            "docker run"
            f" -p {MINIO_PORT}:9000"
            f" -e MINIO_ROOT_USER={MINIO_ROOT_USER}"
            f" -e MINIO_ROOT_PASSWORD={MINIO_ROOT_PASSWORD}"
            f" --name {container_name}"
            " --rm"
            " -d"
            " minio/minio server /data"
        )
        subprocess.check_output(shlex.split(cmd))
        REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
        try:
            yield
        finally:
            subprocess.check_output(["docker", "stop", container_name])

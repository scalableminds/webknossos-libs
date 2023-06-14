import shlex
import subprocess
from pathlib import Path

from upath import UPath

TESTDATA_DIR = Path("testdata")
TESTOUTPUT_DIR = Path("testoutput")


MINIO_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
MINIO_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
MINIO_PORT = "8000"

REMOTE_TESTOUTPUT_DIR = UPath(
    "s3://testoutput",
    key=MINIO_ROOT_USER,
    secret=MINIO_ROOT_PASSWORD,
    client_kwargs={"endpoint_url": f"http://localhost:{MINIO_PORT}"},
)


def start_minio_docker(name: str = "minio") -> None:
    """Minio is an S3 clone and is used as local test server"""
    cmd = (
        "docker run"
        f" -p {MINIO_PORT}:9000"
        f" -e MINIO_ROOT_USER={MINIO_ROOT_USER}"
        f" -e MINIO_ROOT_PASSWORD={MINIO_ROOT_PASSWORD}"
        f" --name {name}"
        " --rm"
        " -d"
        " minio/minio server /data"
    )
    subprocess.check_output(shlex.split(cmd))
    REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)

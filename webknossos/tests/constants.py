from collections.abc import Iterator
from contextlib import contextmanager

from moto.server import ThreadedMotoServer
from upath import UPath

TESTDATA_DIR = UPath(__file__).parent.parent / "testdata"
TESTOUTPUT_DIR = UPath(__file__).parent.parent / "testoutput"


S3_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
S3_PORT = 8000

REMOTE_TESTOUTPUT_DIR = UPath(
    "s3://testoutput",
    key=S3_ROOT_USER,
    secret=S3_ROOT_PASSWORD,
    endpoint_url=f"http://localhost:{S3_PORT}",
)


@contextmanager
def use_moto() -> Iterator[None]:
    """Moto mocks S3 in-process and is used as local test server.

    Unlike minio/rustfs, this runs as a background thread in the same
    process instead of a separate binary/container, so it needs no
    per-OS install and works the same way on Linux, macOS, and Windows.
    """
    server = ThreadedMotoServer(port=S3_PORT)
    server.start()
    try:
        REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
        yield
    finally:
        server.stop()

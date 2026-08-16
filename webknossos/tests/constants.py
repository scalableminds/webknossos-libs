import threading
from collections.abc import Iterator
from contextlib import contextmanager

import waitress
from moto.moto_server.werkzeug_app import (
    DomainDispatcherApplication,
    create_backend_app,
)
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

    Serves moto's WSGI app via waitress instead of moto's own
    ThreadedMotoServer (which uses werkzeug's single-threaded dev server):
    waitress is a production-grade, cross-platform WSGI server and is
    noticeably faster on Windows under the many small S3 requests this
    test suite makes.
    """
    app = DomainDispatcherApplication(create_backend_app)
    server = waitress.create_server(app, host="127.0.0.1", port=S3_PORT, threads=8)

    def run_server() -> None:
        try:
            server.run()
        except OSError:
            # Expected: closing the socket from the main thread interrupts
            # the accept loop's select() call with a benign "bad file
            # descriptor" error.
            pass

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    try:
        REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
        yield
    finally:
        server.close()
        thread.join(timeout=5)

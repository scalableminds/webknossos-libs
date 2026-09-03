import os
import threading
from collections.abc import Generator
from contextlib import contextmanager

import waitress
from moto.moto_server.werkzeug_app import (
    DomainDispatcherApplication,
    create_backend_app,
)
from upath import UPath

# Under pytest-xdist every worker runs in its own process and imports this
# module separately, so deriving the shared paths and the moto port from the
# worker id keeps the workers from clobbering each other. Empty when running
# serially, which keeps the plain `testoutput` name and port 8000.
_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "")
_WORKER_INDEX = int(_WORKER_ID.removeprefix("gw") or 0)

TESTDATA_DIR = UPath(__file__).parent.parent / "testdata"
TESTOUTPUT_DIR = UPath(__file__).parent.parent / f"testoutput{_WORKER_ID}"

TESTOUTPUT_BUCKET = f"testoutput{_WORKER_ID}"

S3_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
S3_PORT = 8000 + _WORKER_INDEX

REMOTE_TESTOUTPUT_DIR = UPath(
    f"s3://{TESTOUTPUT_BUCKET}",
    key=S3_ROOT_USER,
    secret=S3_ROOT_PASSWORD,
    endpoint_url=f"http://localhost:{S3_PORT}",
)


@contextmanager
def use_moto() -> Generator[None]:
    """Moto mocks S3. It is used as local test server, which runs as a
    background thread in the same process.

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
        REMOTE_TESTOUTPUT_DIR.fs.mkdirs(TESTOUTPUT_BUCKET, exist_ok=True)
        yield
    finally:
        server.close()
        thread.join(timeout=5)

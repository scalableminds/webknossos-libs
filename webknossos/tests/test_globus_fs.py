import copy
import json
import os
import shutil
import threading
from collections.abc import Generator
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import globus_sdk
import numpy as np
import pytest
from upath import UPath

import webknossos as wk
from webknossos.dataset.layer.view._array import ArrayException
from webknossos.dataset.transfer_mode import TransferMode
from webknossos.globus_fs import (
    GlobusFileSystem,
    GlobusPath,
    GlobusPermissionError,
    https_scope,
    transfer_scope,
)
from webknossos.utils import copytree

from .constants import REMOTE_TESTOUTPUT_DIR

COLLECTION_ID = "6d1c2a3e-0000-4000-8000-000000000001"
TRANSFER_TOKEN = "transfer-token"
HTTPS_TOKEN = "https-token"


class FakeGlobus:
    """Serves a local directory like a Globus collection, via a fake Transfer API
    and a fake HTTPS server."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[tuple[str, dict[str, str]]] = []
        # Responses to return for the next HTTPS requests before serving the file:
        # an int status, a (status, body) tuple, or "reset" to drop the connection.
        self.https_failures: list[Any] = []
        self.https_token = HTTPS_TOKEN
        self.transfer_status: int | None = None
        self.lock = threading.Lock()

        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                with fake.lock:
                    fake.requests.append((self.path, dict(self.headers)))
                if self.path.startswith("/transfer/"):
                    fake.handle_transfer(self)
                else:
                    fake.handle_https(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def requests_to(self, prefix: str) -> list[tuple[str, dict[str, str]]]:
        return [r for r in self.requests if r[0].startswith(prefix)]

    @staticmethod
    def send(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        handler.send_response(status)
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def send_transfer_error(
        self, handler: BaseHTTPRequestHandler, status: int, code: str
    ) -> None:
        body = {"code": code, "message": code, "request_id": "abc", "resource": "x"}
        self.send(
            handler,
            status,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )

    def handle_transfer(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.headers.get("Authorization") != f"Bearer {TRANSFER_TOKEN}":
            return self.send_transfer_error(handler, 401, "AuthenticationFailed")
        if self.transfer_status is not None:
            return self.send_transfer_error(
                handler, self.transfer_status, "ConsentRequired"
            )
        url = urlparse(handler.path)
        parts = url.path.strip("/").split("/")
        if parts[-1] == "ls":
            query = parse_qs(url.query)
            directory = self.root / query["path"][0].lstrip("/")
            if not directory.exists():
                return self.send_transfer_error(handler, 404, "ClientError.NotFound")
            if not directory.is_dir():
                return self.send_transfer_error(
                    handler, 502, "ExternalError.DirListingFailed.NotDirectory"
                )
            children = sorted(directory.iterdir())
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["100000"])[0])
            page = children[offset : offset + limit]
            data = [
                {
                    "DATA_TYPE": "file",
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size,
                    "last_modified": datetime.fromtimestamp(
                        int(child.stat().st_mtime), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                }
                for child in page
            ]
            body = {
                "DATA_TYPE": "file_list",
                "DATA": data,
                "length": len(data),
                "total": len(children),
                "offset": offset,
                "limit": limit,
            }
        else:
            body = {
                "DATA_TYPE": "endpoint",
                "id": parts[-1],
                "https_server": f"{self.url}/https/",
            }
        self.send(
            handler,
            200,
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )

    def handle_https(self, handler: BaseHTTPRequestHandler) -> None:
        with self.lock:
            failure = self.https_failures.pop(0) if self.https_failures else None
        if failure == "reset":
            handler.close_connection = True
            handler.connection.shutdown(2)
            return
        if failure is not None:
            status, body = failure if isinstance(failure, tuple) else (failure, b"")
            return self.send(handler, status, body)
        if handler.headers.get("Authorization") != f"Bearer {self.https_token}":
            return self.send(handler, 403, b"forbidden")
        path = self.root / unquote(urlparse(handler.path).path).removeprefix("/https/")
        if not path.is_file():
            return self.send(handler, 404, b"not found")
        data = path.read_bytes()
        range_header = handler.headers.get("Range")
        if range_header is None:
            return self.send(handler, 200, data)
        start_str, end_str = range_header.removeprefix("bytes=").split("-")
        start, end = int(start_str), min(int(end_str), len(data) - 1)
        self.send(
            handler,
            206,
            data[start : end + 1],
            {"Content-Range": f"bytes {start}-{end}/{len(data)}"},
        )


@pytest.fixture
def fake_globus(tmp_path: Path) -> Generator[FakeGlobus]:
    root = tmp_path / "collection"
    (root / "data" / "sub").mkdir(parents=True)
    (root / "data" / "a.txt").write_bytes(b"hello globus")
    (root / "data" / "big.bin").write_bytes(bytes(range(256)) * 4096)
    (root / "data" / "sub" / "b.txt").write_bytes(b"b")
    fake = FakeGlobus(root)
    yield fake
    fake.close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("webknossos.globus_fs.time.sleep", lambda _: None)


def make_transfer_client(fake: FakeGlobus) -> globus_sdk.TransferClient:
    return globus_sdk.TransferClient(
        base_url=f"{fake.url}/transfer/",
        authorizer=globus_sdk.AccessTokenAuthorizer(TRANSFER_TOKEN),
        retry_config=globus_sdk.transport.RetryConfig(max_retries=0),
    )


def make_path(fake: FakeGlobus, sub_path: str = "", **kwargs: Any) -> UPath:
    storage_options: dict[str, Any] = {
        "transfer_client": make_transfer_client(fake),
        "https_authorizer": globus_sdk.AccessTokenAuthorizer(HTTPS_TOKEN),
        "skip_instance_cache": True,
    }
    storage_options.update(kwargs)
    return UPath(f"globus://{COLLECTION_ID}/{sub_path}", **storage_options)


def test_upath_resolution() -> None:
    path = UPath(f"globus://{COLLECTION_ID}/data/ds")
    assert isinstance(path, GlobusPath)
    assert path.protocol == "globus"
    assert path.parts == (f"{COLLECTION_ID}/", "data", "ds")
    assert str(path.parent) == f"globus://{COLLECTION_ID}/data"
    assert str(path / "color" / "1") == f"globus://{COLLECTION_ID}/data/ds/color/1"
    assert isinstance(path.fs, GlobusFileSystem)


def test_scopes() -> None:
    assert https_scope("abc") == "https://auth.globus.org/scopes/abc/https"
    assert transfer_scope("abc") == (
        "urn:globus:auth:scope:transfer.api.globus.org:all"
        "[*https://auth.globus.org/scopes/abc/data_access]"
    )


def test_ls_and_info(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data")
    assert sorted(p.name for p in path.iterdir()) == ["a.txt", "big.bin", "sub"]
    assert path.is_dir()
    assert (path / "sub").is_dir()
    assert (path / "a.txt").is_file()

    info = path.fs.info(f"{COLLECTION_ID}/data/a.txt")
    assert info["type"] == "file"
    assert info["size"] == len(b"hello globus")
    assert isinstance(info["mtime"], float)
    assert (path / "a.txt").stat().st_size == len(b"hello globus")

    root_info = path.fs.info(f"globus://{COLLECTION_ID}/")
    assert root_info["type"] == "directory"

    assert not (path / "missing.txt").exists()
    assert not (path / "missing" / "x").exists()
    with pytest.raises(FileNotFoundError):
        path.fs.ls(f"{COLLECTION_ID}/missing")
    # ls of a file returns its info
    assert path.fs.ls(f"{COLLECTION_ID}/data/a.txt", detail=False) == [
        f"{COLLECTION_ID}/data/a.txt"
    ]


def test_ls_pagination_and_cache(fake_globus: FakeGlobus) -> None:
    for i in range(5):
        (fake_globus.root / "data" / "sub" / f"file{i}").write_bytes(b"x")
    path = make_path(fake_globus, "data/sub", ls_page_size=2)
    names = path.fs.ls(str(path), detail=False)
    assert len(names) == 6
    ls_requests = [r for r in fake_globus.requests_to("/transfer/") if "/ls?" in r[0]]
    assert len(ls_requests) == 4  # pages with 2, 2, 2 and 0 entries
    assert all(r[1]["Authorization"] == f"Bearer {TRANSFER_TOKEN}" for r in ls_requests)

    path.fs.ls(str(path))
    (path / "file0").stat()
    assert len(fake_globus.requests_to("/transfer/")) == 4


def test_read_whole_and_ranged(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data")
    assert (path / "a.txt").read_bytes() == b"hello globus"
    # https_server is looked up once via the Transfer API
    endpoint_requests = [
        r for r in fake_globus.requests_to("/transfer/") if "/ls?" not in r[0]
    ]
    assert len(endpoint_requests) == 1

    big = (fake_globus.root / "data" / "big.bin").read_bytes()
    fake_globus.requests.clear()
    with path.fs.open(f"{COLLECTION_ID}/data/big.bin", "rb", block_size=100_000) as f:
        f.seek(500_000)
        assert f.read(10) == big[500_000:500_010]
        f.seek(5)
        assert f.read(3) == big[5:8]
    https_requests = fake_globus.requests_to("/https/")
    assert [r[1]["Range"] for r in https_requests] == [
        "bytes=500000-600009",
        "bytes=5-100007",
    ]
    assert all(r[1]["Authorization"] == f"Bearer {HTTPS_TOKEN}" for r in https_requests)

    assert path.fs.cat_file(f"{COLLECTION_ID}/data/big.bin", start=-4) == big[-4:]


def test_copyfileobj_streams_in_blocks(fake_globus: FakeGlobus, tmp_path: Path) -> None:
    path = make_path(fake_globus, "data/big.bin")
    with path.fs.open(path.path, "rb", block_size=200_000) as src:
        with open(tmp_path / "out.bin", "wb") as dst:
            shutil.copyfileobj(src, dst)
    assert (tmp_path / "out.bin").read_bytes() == (
        fake_globus.root / "data" / "big.bin"
    ).read_bytes()
    ranges = [r[1]["Range"] for r in fake_globus.requests_to("/https/")]
    assert len(ranges) > 1
    for r in ranges:
        start, end = map(int, r.removeprefix("bytes=").split("-"))
        assert end - start < 300_000


def test_retries_transient_errors(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data/a.txt")
    fake_globus.https_failures = [
        503,
        "reset",
        (404, b'{"code": "ENDPOINT_ERROR"}'),
        500,
    ]
    assert path.read_bytes() == b"hello globus"
    assert fake_globus.https_failures == []


def test_retries_exhausted(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data/a.txt", retries=2)
    fake_globus.https_failures = [503, 503, 503, 503]
    with pytest.raises(OSError, match="503"):
        path.read_bytes()
    assert len(fake_globus.https_failures) == 1


def test_https_error_mapping(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data/a.txt")
    fake_globus.https_failures = [(404, b"not found")]
    with pytest.raises(FileNotFoundError):
        path.read_bytes()

    fake_globus.https_token = "other"
    with pytest.raises(GlobusPermissionError, match=https_scope(COLLECTION_ID)):
        path.read_bytes()


def test_transfer_error_mapping(fake_globus: FakeGlobus) -> None:
    fake_globus.transfer_status = 403
    path = make_path(fake_globus, "data")
    with pytest.raises(GlobusPermissionError) as exc_info:
        list(path.iterdir())
    assert COLLECTION_ID in str(exc_info.value)
    assert transfer_scope(COLLECTION_ID) in str(exc_info.value)


def test_token_callables_and_refresh(fake_globus: FakeGlobus) -> None:
    class RefreshingAuthorizer(globus_sdk.authorizers.GlobusAuthorizer):
        def __init__(self) -> None:
            self.token = "expired"

        def get_authorization_header(self) -> str:
            return f"Bearer {self.token}"

        def handle_missing_authorization(self) -> bool:
            self.token = HTTPS_TOKEN
            return True

    path = make_path(
        fake_globus,
        "data/a.txt",
        transfer_client=None,
        transfer_authorizer=lambda: TRANSFER_TOKEN,
        https_authorizer=RefreshingAuthorizer(),
        https_server=f"{fake_globus.url}/https",
    )
    # The default client talks to the real Transfer API, so only test the header here.
    header = path.fs._transfer_client.authorizer.get_authorization_header()
    assert header == f"Bearer {TRANSFER_TOKEN}"
    path.fs._transfer_client = make_transfer_client(fake_globus)

    fake_globus.https_failures = [401]
    assert path.read_bytes() == b"hello globus"

    path = make_path(fake_globus, "data/a.txt", https_authorizer=lambda: HTTPS_TOKEN)
    assert path.read_bytes() == b"hello globus"


def test_read_only(fake_globus: FakeGlobus) -> None:
    path = make_path(fake_globus, "data")
    with pytest.raises(NotImplementedError, match="read-only"):
        (path / "new.txt").write_bytes(b"x")
    with pytest.raises(NotImplementedError, match="read-only"):
        (path / "new").mkdir()
    with pytest.raises(NotImplementedError, match="read-only"):
        (path / "a.txt").unlink()
    with pytest.raises(NotImplementedError, match="read-only"):
        path.fs.rm(str(path), recursive=True)
    with pytest.raises(NotImplementedError, match="read-only"):
        path.fs.mv(f"{path.path}/a.txt", f"{path.path}/c.txt")


def _create_dataset(path: Path) -> np.ndarray:
    ds = wk.Dataset(path, voxel_size=(11, 11, 24))
    layer = ds.add_layer(
        "color",
        wk.COLOR_CATEGORY,
        dtype="uint8",
        bounding_box=wk.BoundingBox((0, 0, 0), (32, 32, 16)),
    )
    data = np.random.default_rng(0).integers(0, 255, (32, 32, 16), dtype=np.uint8)
    layer.add_mag(1).write(data, absolute_offset=(0, 0, 0))
    return data


def test_open_and_copy_dataset(fake_globus: FakeGlobus, tmp_path: Path) -> None:
    data = _create_dataset(fake_globus.root / "datasets" / "my_ds")
    path = make_path(fake_globus, "datasets/my_ds")

    ds = wk.Dataset.open(path)
    assert ds.voxel_size == (11, 11, 24)
    assert ds.get_layer("color").dtype == np.uint8
    assert list(ds.get_layer("color").mags) == [wk.Mag(1)]
    with pytest.raises(ArrayException) as exc_info:
        ds.get_layer("color").get_mag(1).read()
    assert "Copy the data first" in str(exc_info.value.__cause__)

    copytree(path, UPath(tmp_path / "copy"))
    copied = wk.Dataset.open(tmp_path / "copy")
    np.testing.assert_array_equal(copied.get_layer("color").get_mag(1).read()[0], data)


def test_transfer_dataset_items(fake_globus: FakeGlobus, tmp_path: Path) -> None:
    data = _create_dataset(fake_globus.root / "datasets" / "my_ds")
    ds = wk.Dataset.open(make_path(fake_globus, "datasets/my_ds"))

    # Paths as returned by WEBKNOSSOS in publish_to_preliminary_dataset
    data_source = copy.deepcopy(ds._properties)
    for layer in data_source.data_layers:
        for mag in layer.mags:
            mag.path = str(
                tmp_path / "published" / layer.name / mag.mag.to_layer_name()
            )
    ds._transfer_dataset_items(data_source, TransferMode.COPY)

    src_mag = fake_globus.root / "datasets" / "my_ds" / "color" / "1"
    dst_mag = tmp_path / "published" / "color" / "1"
    src_files = sorted(
        p.relative_to(src_mag) for p in src_mag.rglob("*") if p.is_file()
    )
    assert src_files
    assert src_files == sorted(
        p.relative_to(dst_mag) for p in dst_mag.rglob("*") if p.is_file()
    )
    for f in src_files:
        assert (src_mag / f).read_bytes() == (dst_mag / f).read_bytes()
    del data


def test_resume_copy(fake_globus: FakeGlobus, tmp_path: Path) -> None:
    _create_dataset(fake_globus.root / "datasets" / "my_ds")
    path = make_path(fake_globus, "datasets/my_ds")
    copytree(path, UPath(tmp_path / "copy"))

    fake_globus.requests.clear()
    copytree(path, UPath(tmp_path / "copy"), skip_existing_same_size=True)
    assert fake_globus.requests_to("/https/") == []


@pytest.mark.usefixtures("moto_server")
def test_copy_dataset_to_s3(fake_globus: FakeGlobus) -> None:
    data = _create_dataset(fake_globus.root / "datasets" / "my_ds")
    path = make_path(fake_globus, "datasets/my_ds")
    out_path = REMOTE_TESTOUTPUT_DIR / "globus_copy"

    copytree(path, out_path)
    copied = wk.Dataset.open(out_path)
    np.testing.assert_array_equal(copied.get_layer("color").get_mag(1).read()[0], data)


@pytest.mark.skipif(
    not all(
        os.environ.get(var)
        for var in (
            "WK_GLOBUS_TEST_COLLECTION",
            "WK_GLOBUS_TEST_PATH",
            "WK_GLOBUS_TRANSFER_TOKEN",
            "WK_GLOBUS_HTTPS_TOKEN",
        )
    ),
    reason="Requires access to a real Globus collection.",
)
def test_live_collection(tmp_path: Path) -> None:
    """Copies a dataset from a real Globus collection, set via env vars:
    WK_GLOBUS_TEST_COLLECTION (collection id), WK_GLOBUS_TEST_PATH (absolute dataset path),
    WK_GLOBUS_TRANSFER_TOKEN and WK_GLOBUS_HTTPS_TOKEN (access tokens)."""
    path = UPath(
        f"globus://{os.environ['WK_GLOBUS_TEST_COLLECTION']}/{os.environ['WK_GLOBUS_TEST_PATH'].lstrip('/')}",
        transfer_authorizer=globus_sdk.AccessTokenAuthorizer(
            os.environ["WK_GLOBUS_TRANSFER_TOKEN"]
        ),
        https_authorizer=globus_sdk.AccessTokenAuthorizer(
            os.environ["WK_GLOBUS_HTTPS_TOKEN"]
        ),
    )
    ds = wk.Dataset.open(path)
    assert len(ds.layers) > 0
    copytree(path, UPath(tmp_path / "copy"))
    wk.Dataset.open(tmp_path / "copy")

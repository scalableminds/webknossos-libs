from collections import defaultdict
from pathlib import Path

import httpx
from upath import UPath

from webknossos.client._resumable import Resumable
from webknossos.client._resumable.file import ResumableFile


def _parse_multipart(request: httpx.Request) -> dict[str, bytes]:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields = {}
    for part in request.read().split(b"--" + boundary)[1:-1]:
        headers, _, value = part.partition(b"\r\n\r\n")
        name = headers.split(b'name="')[1].split(b'"')[0].decode()
        fields[name] = value.removesuffix(b"\r\n")
    return fields


def test_resumable_opens_files_lazily(tmp_path: Path) -> None:
    simultaneous_uploads = 2
    file_contents = {f"file_{i}.bin": bytes([i]) * (2500 + i) for i in range(50)}
    for name, content in file_contents.items():
        (tmp_path / name).write_bytes(content)

    files: list[ResumableFile] = []
    received_chunks: dict[str, dict[int, bytes]] = defaultdict(dict)
    max_open_files = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal max_open_files
        max_open_files = max(
            max_open_files, sum(file._fp is not None for file in files)
        )
        fields = _parse_multipart(request)
        filename = fields["resumableFilename"].decode()
        chunk_number = int(fields["resumableChunkNumber"])
        received_chunks[filename][chunk_number] = fields["file"]
        return httpx.Response(200)

    with Resumable(
        "http://test/upload",
        chunk_size=1000,
        simultaneous_uploads=simultaneous_uploads,
        test_chunks=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as session:
        files.extend(
            session.add_file(UPath(tmp_path / name), UPath(name))
            for name in file_contents
        )

    assert 0 < max_open_files <= simultaneous_uploads
    for file in files:
        assert file.is_completed
        assert file._fp is None

    for name, content in file_contents.items():
        chunks = received_chunks[name]
        assert b"".join(chunks[i] for i in sorted(chunks)) == content

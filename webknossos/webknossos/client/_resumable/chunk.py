import mimetypes
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from .file import FileChunk, ResumableFile
from .util import Config

RETRY_DELAY = 5  # seconds


class ResumableError(Exception):
    pass


def resolve_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> None:
    """Make sure a chunk is uploaded to the server and mark it as completed.

    Args:
        client: A client to use for communication with the server
        config: The configuration of the resumable client
        file: The parent file of the chunk to be resolved
        chunk: The chunk to be resolved
    """

    exists_on_server = False
    if config.test_chunks:
        exists_on_server = _test_chunk(client, config, file, chunk)

    if not exists_on_server:
        tries = 0
        while not _send_chunk(client, config, file, chunk):
            time.sleep(RETRY_DELAY)
            tries += 1
            if tries >= config.max_chunk_retries:
                raise ResumableError("max retries exceeded")

    file.mark_chunk_completed(chunk)


def _test_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> bool:
    """Check if the chunk exists on the server.

    Args:
        client: The HTTP client
        config: The resumable configuration
        file: The parent file
        chunk: The chunk to test

    Returns:
        bool: True if the chunk exists on the server
    """
    response = client.get(
        config.target, params=_build_query(file, chunk, config.additional_query_params)
    )
    return response.status_code == 200


def _send_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> bool:
    """Upload the chunk to the server.

    Args:
        client: The HTTP client instance
        config: The resumable configuration settings
        file: The parent file of the chunk
        chunk: The chunk to upload

    Returns:
        bool: True if the upload was successful

    Raises:
        ResumableError: If the server responded with an error code indicating permanent failure
    """
    try:
        response = client.post(
            config.target,
            data=_build_query(file, chunk, config.additional_query_params),
            files={"file": chunk.read()},
        )
    except httpx.TransportError:
        # The request itself failed. The calling function will retry.
        return False
    if response.status_code in config.permanent_errors:
        raise ResumableError(f"{response.status_code} Error: {response.text}")
    return response.status_code in [200, 201]


def _build_query(
    file: ResumableFile, chunk: FileChunk, additional_query_params: Dict[str, str]
) -> Dict[str, Any]:
    """Build the query parameters for a chunk test or upload."""
    return {
        "resumableChunkSize": file.chunk_size,
        "resumableTotalSize": file.size,
        "resumableType": _file_type(file.path),
        "resumableIdentifier": str(file.unique_identifier),
        "resumableFilename": file.path.name,
        "resumableRelativePath": str(file.relative_path),
        "resumableTotalChunks": len(file.chunks),
        "resumableChunkNumber": chunk.index + 1,
        "resumableCurrentChunkSize": chunk.size,
        **additional_query_params,
    }


def _file_type(path: Path) -> str:
    """Mimic the type parameter of a JS File object.

    Resumable.js uses the File object's type attribute to guess mime type,
    which is guessed from file extension according to
    https://developer.mozilla.org/en-US/docs/Web/API/File/type.

    Args:
        path: The path to guess the mime type of

    Returns:
        str: The inferred mime type, or '' if none could be inferred
    """
    type_, _ = mimetypes.guess_type(path.name)
    # When no type can be inferred, File.type returns an empty string
    return "" if type_ is None else type_

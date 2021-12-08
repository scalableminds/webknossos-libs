import mimetypes
from pathlib import Path
from typing import Any, Dict

import httpx

from .file import FileChunk, ResumableFile
from .util import Config


class ResumableError(Exception):
    pass


def resolve_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> None:
    """Make sure a chunk is uploaded to the server and mark it as completed.

    Parameters
    ----------
    client : httpx.Client
        A client to use for communication with the server
    config : resumable.util.Config
        The configuration of the resumable client
    file : resumable.file.ResumableFile
        The parent file of the chunk to be resolved
    chunk : resumable.file.FileChunk
        The chunk to be resolved
    """

    exists_on_server = False
    if config.test_chunks:
        exists_on_server = _test_chunk(client, config, file, chunk)

    if not exists_on_server:
        tries = 0
        while not _send_chunk(client, config, file, chunk):
            tries += 1
            if tries >= config.max_chunk_retries:
                raise ResumableError("max retries exceeded")

    file.mark_chunk_completed(chunk)


def _test_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> bool:
    """Check if the chunk exists on the server.

    Returns
    -------
    bool
        True if the chunk exists on the server
    """
    response = client.get(
        config.target, params=_build_query(file, chunk, config.additional_query_params)
    )
    return response.status_code == 200


def _send_chunk(
    client: httpx.Client, config: Config, file: ResumableFile, chunk: FileChunk
) -> bool:
    """Upload the chunk to the server.

    Returns
    -------
    bool
        True if the upload was successful

    Raises
    ------
    ResumableError
        If the server responded with an error code indicating permanent failure
    """
    response = client.post(
        config.target,
        data=_build_query(file, chunk, config.additional_query_params),
        files={"file": chunk.read()},
    )
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
    which is guessed from file extention accoring to
    https://developer.mozilla.org/en-US/docs/Web/API/File/type.

    Parameters
    ----------
    path : str
        The path to guess the mime type of

    Returns
    -------
    str
        The inferred mime type, or '' if none could be inferred
    """
    type_, _ = mimetypes.guess_type(path.name)
    # When no type can be inferred, File.type returns an empty string
    return "" if type_ is None else type_

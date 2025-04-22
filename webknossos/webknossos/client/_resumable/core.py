import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import httpx

from .chunk import resolve_chunk
from .file import ResumableFile
from .util import CallbackDispatcher, Config

MiB = 1024 * 1024


class Resumable:
    """A resumable-httpx upload client.

    Args:
        target: The URL of the resumable upload target
        chunk_size: The size, in bytes, of file chunks to be uploaded. Defaults to 1MB.
        simultaneous_uploads: The number of file chunk uploads to attempt at once.
            Defaults to 3.
        headers: A dictionary of additional HTTP headers to include in requests.
            Defaults to None.
        test_chunks: Flag indicating if the client should check with the server if a chunk
            already exists with a GET request prior to attempting to upload the chunk with
            a POST. Defaults to True.
        max_chunk_retries: The number of times to retry uploading a chunk.
            Defaults to 100.
        permanent_errors: HTTP status codes that indicate the upload of a chunk has failed
            and should not be retried. Defaults to (400, 404, 415, 500, 501).

    Attributes:
        file_added: Triggered when a file has been added, passing the file object.
        file_completed: Triggered when a file upload has completed, passing the file object.
        chunk_completed: Triggered when a chunk upload has completed, passing the file and
            chunk objects.
    """

    def __init__(
        self,
        target: str,
        chunk_size: int = MiB,
        simultaneous_uploads: int = 3,
        headers: dict[str, Any] | None = None,
        test_chunks: bool = True,
        max_chunk_retries: int = 100,
        permanent_errors: Sequence[int] = (400, 404, 415, 500, 501),
        query: dict[str, Any] | None = None,
        generate_unique_identifier: Callable[[Path, Path], str] = lambda _path,
        _relative_path: str(uuid.uuid4()),
        client: httpx.Client = httpx.Client(),
    ) -> None:
        if headers is None:
            headers = {}
        if query is None:
            query = {}

        self.config = Config(
            target=target,
            chunk_size=chunk_size,
            simultaneous_uploads=simultaneous_uploads,
            headers=headers,
            test_chunks=test_chunks,
            max_chunk_retries=max_chunk_retries,
            permanent_errors=permanent_errors,
            additional_query_params=query,
            generate_unique_identifier=generate_unique_identifier,
        )

        self.client = client
        self.client.headers.update(headers)

        self.files: list[ResumableFile] = []

        self.executor = ThreadPoolExecutor(simultaneous_uploads)
        self.futures: list[Future] = []

        self.file_added = CallbackDispatcher()
        self.file_completed = CallbackDispatcher()
        self.chunk_completed = CallbackDispatcher()

    def add_file(
        self, path: PathLike | str, relative_path: Path | None
    ) -> ResumableFile:
        file = ResumableFile(
            Path(path),
            relative_path,
            self.config.chunk_size,
            self.config.generate_unique_identifier,
        )
        self.files.append(file)

        self.file_added.trigger(file)
        file.completed.register(partial(self.file_completed.trigger, file))
        file.chunk_completed.register(partial(self.chunk_completed.trigger, file))

        for chunk in file.chunks:
            future = self.executor.submit(
                resolve_chunk, self.client, self.config, file, chunk
            )
            self.futures.append(future)

        return file

    def _wait(self) -> None:
        """Wait until all current uploads are completed."""
        for future in as_completed(self.futures):
            exception = future.exception()
            if exception is not None:
                raise exception

    def _cancel_remaining_futures(self) -> None:
        for future in self.futures:
            if not future.done():
                future.cancel()

    def join(self) -> None:
        """Block until all uploads are complete, or an error occurs."""
        try:
            self._wait()
        except:  # noqa: E722
            self._cancel_remaining_futures()
            raise
        finally:
            self.executor.shutdown()
            for file in self.files:
                file.close()

    def __enter__(self) -> "Resumable":
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.join()

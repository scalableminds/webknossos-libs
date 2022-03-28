import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import httpx

from .chunk import resolve_chunk
from .file import ResumableFile
from .util import CallbackDispatcher, Config

MiB = 1024 * 1024


class Resumable:
    """A resumable-httpx upload client.

    Parameters
    ----------
    target : str
        The URL of the resumable upload target
    chunk_size : int, optional
        The size, in bytes, of file chunks to be uploaded
    simultaneous_uploads : int, optional
        The number of file chunk uploads to attempt at once
    headers : dict, optional
        A dictionary of additional HTTP headers to include in requests
    test_chunks : bool
        Flag indicating if the client should check with the server if a chunk
        already exists with a GET request prior to attempting to upload the
        chunk with a POST
    max_chunk_retries : int, optional
        The number of times to retry uploading a chunk
    permanent_errors : collection of int, optional
        HTTP status codes that indicate the upload of a chunk has failed and
        should not be retried

    Attributes
    ----------
    file_added : resumable.util.CallbackDispatcher
        Triggered when a file has been added, passing the file object
    file_completed : resumable.util.CallbackDispatcher
        Triggered when a file upload has completed, passing the file object
    chunk_completed : resumable.util.CallbackDispatcher
        Triggered when a chunk upload has completed, passing the file and chunk
        objects
    """

    def __init__(
        self,
        target: str,
        chunk_size: int = MiB,
        simultaneous_uploads: int = 3,
        headers: Optional[Dict[str, Any]] = None,
        test_chunks: bool = True,
        max_chunk_retries: int = 100,
        permanent_errors: Sequence[int] = (400, 404, 415, 500, 501),
        query: Optional[Dict[str, Any]] = None,
        generate_unique_identifier: Callable[
            [Path, Path], str
        ] = lambda _path, _relative_path: str(uuid.uuid4()),
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

        self.files: List[ResumableFile] = []

        self.executor = ThreadPoolExecutor(simultaneous_uploads)
        self.futures: List[Future] = []

        self.file_added = CallbackDispatcher()
        self.file_completed = CallbackDispatcher()
        self.chunk_completed = CallbackDispatcher()

    def add_file(
        self, path: Union[PathLike, str], relative_path: Optional[Path]
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

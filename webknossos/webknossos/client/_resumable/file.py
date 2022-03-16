from __future__ import division

from collections import namedtuple
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Callable, List, Optional

from .util import CallbackDispatcher

FileChunk = namedtuple("FileChunk", ["index", "size", "read"])


def build_chunks(
    read_bytes: Callable[[int, int], bytes], file_size: int, chunk_size: int
) -> List[FileChunk]:
    chunks = []

    index = 0
    start = 0

    while start < file_size:
        end = min(start + chunk_size, file_size)
        size = end - start

        chunk = FileChunk(index, size, partial(read_bytes, start, size))
        chunks.append(chunk)

        index += 1
        start += chunk_size

    return chunks


class ResumableFile:
    """A file to be uploaded in a resumable session.

    Parameters
    ----------
    path : str or pathlib.Path
        The path of the file
    relative_path : pathlib.Path or None
        The relative path of the file
    chunk_size : int
        The size, in bytes, of chunks uploaded in a single request

    Attributes
    ----------
    completed : resumable.util.CallbackDispatcher
        Triggered when all chunks of the file have been uploaded
    chunk_completed : resumable.util.CallbackDispatcher
        Triggered when a chunks of the file has been uploaded, passing the
        chunk
    """

    def __init__(
        self,
        path: Path,
        relative_path: Optional[Path],
        chunk_size: int,
        generate_unique_identifier: Callable[[Path, Path], str],
    ) -> None:

        self.path = path
        if relative_path is not None:
            self.relative_path = relative_path
        else:
            self.relative_path = path
        self.unique_identifier = generate_unique_identifier(
            self.path, self.relative_path
        )
        self.chunk_size = chunk_size
        self.size = self.path.stat().st_size

        self._fp = open(self.path, "rb")
        self._fp_lock = Lock()

        self.chunks = build_chunks(self._read_bytes, self.size, chunk_size)
        self._chunk_done = {chunk: False for chunk in self.chunks}

        self.completed = CallbackDispatcher()
        self.chunk_completed = CallbackDispatcher()

    def close(self) -> None:
        """Close the file."""
        self._fp.close()

    def _read_bytes(self, start: int, num_bytes: int) -> bytes:
        """Read a byte range from the file."""
        with self._fp_lock:
            self._fp.seek(start)
            return self._fp.read(num_bytes)

    @property
    def is_completed(self) -> bool:
        """Indicates if all chunks of this file have been uploaded."""
        return all(self._chunk_done.values())

    @property
    def fraction_completed(self) -> float:
        """The fraction of the file that has been completed."""
        return sum(self._chunk_done.values()) / len(self.chunks)

    def mark_chunk_completed(self, chunk: FileChunk) -> None:
        """Mark a chunk of this file as having been successfully uploaded.

        If all chunks have been completed, this will trigger the `completed`
        callback of this file.

        Parameters
        ----------
        chunk : resumable.chunk.FileChunk
            The chunk to mark as completed
        """
        self._chunk_done[chunk] = True
        if self.is_completed:
            self.completed.trigger()
            self.close()
        self.chunk_completed.trigger(chunk)

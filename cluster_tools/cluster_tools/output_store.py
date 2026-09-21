import contextlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable


class OutputStore(ABC):
    """Persists the pickled outputs of jobs and hands them back to the executor.

    A job's output is the pickled tuple `(success, result_or_traceback)`. Successful
    outputs serve as checkpoints, so implementations must keep them apart from failed ones
    (e.g. under a different key), but `poll`/`read` must report both.

    Instances are pickled into the job processes, so they must be picklable and must not
    rely on state that is local to the submitting process.
    """

    @abstractmethod
    def default_key(self, job_id: str) -> str:
        """Key for a job without a custom output key."""

    @abstractmethod
    def write(self, key: str, data: bytes, *, success: bool) -> None:
        """Called in the job process. Must be atomic w.r.t. `poll`."""

    @abstractmethod
    def poll(self, keys: Iterable[str]) -> set[str]:
        """Returns the subset of `keys` for which an output was written."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Removes the output (successful or failed) of `key`, if any."""


class FileOutputStore(OutputStore):
    """Stores outputs as files. Keys are file paths.

    Successful outputs are written to `<key>`, failed ones to `<key>.preliminary`.
    """

    def __init__(self, directory: str | os.PathLike | None = None):
        self.directory = None if directory is None else str(directory)

    def default_key(self, job_id: str) -> str:
        assert self.directory is not None, (
            "FileOutputStore needs a directory to derive default keys."
        )
        return os.path.join(self.directory, f"cfut.out.{job_id}.pickle")

    @staticmethod
    def preliminary_path(key: str) -> str:
        return f"{key}.preliminary"

    def write(self, key: str, data: bytes, *, success: bool) -> None:
        dest = key if success else self.preliminary_path(key)
        tmp = f"{dest}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, dest)

    def poll(self, keys: Iterable[str]) -> set[str]:
        return {
            key
            for key in keys
            if os.path.exists(key) or os.path.exists(self.preliminary_path(key))
        }

    def read(self, key: str) -> bytes:
        path = key if os.path.exists(key) else self.preliminary_path(key)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        for path in (key, self.preliminary_path(key)):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

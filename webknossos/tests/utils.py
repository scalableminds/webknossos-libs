import uuid
from collections.abc import Generator
from contextlib import contextmanager

from upath import UPath


@contextmanager
def TestTemporaryDirectoryNonLocal() -> Generator[UPath, None, None]:
    """Gives a temporary directory as UPath which does not use the "local" protocol (local file system).
    Useful for testing functionality that uses non-local UPaths.
    Currently implemented to use an in-memory file system. (no persistence across lifetime of the process)."""
    random_prefix = str(uuid.uuid4())
    temp_dir = UPath(f"memory:///{random_prefix}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir

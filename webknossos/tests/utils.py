import uuid
from collections.abc import Generator
from contextlib import contextmanager
from shutil import copy
from tempfile import NamedTemporaryFile

import httpx
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


IMS_FIXTURE_URL = "https://static.webknossos.org/data/wklibs-samples/brain_crop3.ims"


def download_ims_fixture(tmp_upath: UPath) -> UPath:
    """Downloads the brain_crop3.ims test fixture (2 channels, single
    timepoint, uint16) to tmp_upath and returns its path."""
    ims_path = tmp_upath / "brain_crop3.ims"
    with NamedTemporaryFile() as download_file:
        with httpx.stream("GET", IMS_FIXTURE_URL, follow_redirects=True) as response:
            for chunk in response.iter_bytes():
                download_file.write(chunk)
        # copy() reopens the file by name, so pending writes must be flushed
        # to disk first, or the copy would see a truncated file.
        download_file.flush()
        copy(download_file.name, str(ims_path))
    return ims_path

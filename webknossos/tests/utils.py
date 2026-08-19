import importlib.util
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.typing import DTypeLike
from upath import UPath

# pylibCZIrw ships no wheel past cp313 (mirrors pyproject.toml's czi extra,
# `pylibCZIrw ==5.1.1; python_version < '3.14'`), so it is expected to be
# missing on newer Pythons and tests that need it skip there. On earlier
# versions it must be installed; if it's missing there instead, that's a
# broken test environment, not a reason to skip, so such tests are left to
# fail rather than silently pass over pylibCZIrw-only code paths.
PYLIBCZIRW_EXPECTED = sys.version_info < (3, 14)
HAS_PYLIBCZIRW = importlib.util.find_spec("pylibCZIrw") is not None
requires_pylibczirw = pytest.mark.skipif(
    not HAS_PYLIBCZIRW and not PYLIBCZIRW_EXPECTED,
    reason="pylibCZIrw is not installed for this Python version",
)


@contextmanager
def TestTemporaryDirectoryNonLocal() -> Generator[UPath, None, None]:
    """Gives a temporary directory as UPath which does not use the "local" protocol (local file system).
    Useful for testing functionality that uses non-local UPaths.
    Currently implemented to use an in-memory file system. (no persistence across lifetime of the process)."""
    random_prefix = str(uuid.uuid4())
    temp_dir = UPath(f"memory:///{random_prefix}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir

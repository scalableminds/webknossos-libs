import importlib.util
import uuid
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.typing import DTypeLike
from upath import UPath

# pylibCZIrw ships no wheel past cp313, so the czi extra is uninstalled on
# newer Pythons; tests that need it skip instead of failing.
HAS_PYLIBCZIRW = importlib.util.find_spec("pylibCZIrw") is not None
requires_pylibczirw = pytest.mark.skipif(
    not HAS_PYLIBCZIRW,
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


def create_synthetic_czi(
    path: UPath,
    *,
    num_timepoints: int = 1,
    num_czi_channels: int = 1,
    z: int = 1,
    y: int = 8,
    x: int = 10,
    dtype: DTypeLike = np.uint16,
) -> np.ndarray:
    """Writes a real .czi via pylibCZIrw and returns the data as
    (t, czi_channel, z, y, x), so tests can compare against it directly.

    Unlike the .ims helper this produces a genuine file rather than a stand-in,
    so no monkeypatching is needed. Voxel values are unique per position and
    per (t, czi_channel), which is what makes a mixed-up axis or a wrongly
    offset roi visible rather than merely plausible.
    """
    from pylibCZIrw import czi as pyczi

    data = np.arange(num_timepoints * num_czi_channels * z * y * x).reshape(
        num_timepoints, num_czi_channels, z, y, x
    )
    data = (data % np.iinfo(np.uint16).max).astype(dtype)

    with pyczi.create_czi(str(path)) as writer:
        for t in range(num_timepoints):
            for c in range(num_czi_channels):
                for k in range(z):
                    writer.write(
                        # pylibCZIrw writes planes as (y, x, components)
                        data[t, c, k][:, :, np.newaxis],
                        plane={"T": t, "C": c, "Z": k},
                    )
    return data

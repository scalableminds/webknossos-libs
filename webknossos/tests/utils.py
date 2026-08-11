import uuid
from collections.abc import Generator
from contextlib import contextmanager

import h5py
import httpx
import numpy as np
from numpy.typing import DTypeLike
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
    # Streamed straight to the destination rather than via a NamedTemporaryFile:
    # on Windows that keeps the file open exclusively, so copying it by name
    # fails with PermissionError.
    ims_path = tmp_upath / "brain_crop3.ims"
    with (
        httpx.stream("GET", IMS_FIXTURE_URL, follow_redirects=True) as response,
        ims_path.open("wb") as out_file,
    ):
        for chunk in response.iter_bytes():
            out_file.write(chunk)
    return ims_path


def create_synthetic_multi_timepoint_ims(
    path: UPath,
    *,
    num_timepoints: int,
    num_channels: int,
    z: int,
    y: int,
    x: int,
    dtype: DTypeLike = np.uint16,
) -> None:
    """Writes a minimal HDF5 structure matching what ImsImageSource actually
    reads (DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data). This
    intentionally skips the DataSetInfo attributes that the full
    imaris_ims_file_reader library needs, since callers monkeypatch
    ims_image_source._read_ims_metadata_quietly instead of relying on a
    byte-perfect Imaris file.

    `dtype` matters for multi-channel files: only three uint8 channels are
    written into a single layer, everything else is split per channel."""
    with h5py.File(str(path), "w") as f:
        res0 = f.create_group("DataSet").create_group("ResolutionLevel 0")
        for t in range(num_timepoints):
            tp = res0.create_group(f"TimePoint {t}")
            for c in range(num_channels):
                ch = tp.create_group(f"Channel {c}")
                # encodes (t, c) into every voxel so tests can verify both
                # axes were read correctly, independent of x/y/z position
                data = np.full((z, y, x), t * 100 + c, dtype=dtype)
                ch.create_dataset("Data", data=data)


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

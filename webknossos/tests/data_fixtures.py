"""Provides test data fixtures: downloaded (mainly from the `wklibs-samples`
bucket on static.webknossos.org) or synthetically generated.

Downloads are cached under CACHE_DIR so repeated local test runs reuse
what's already there instead of re-fetching it every time. The cache is
gitignored (via the repo's generic `.cache` rule) and never cleaned up
automatically; delete it by hand if you want to force a re-download."""

import os
import stat
import uuid
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import h5py
import httpx
import numpy as np
from upath import UPath

from webknossos.utils import rmtree

CACHE_DIR = UPath(__file__).parent.parent / ".cache" / "wklibs-samples"

WKLIBS_SAMPLES_BASE_URL = "https://static.webknossos.org/data/wklibs-samples"


def _tmp_cache_path(name: str) -> UPath:
    return CACHE_DIR / f".tmp-{name}-{uuid.uuid4().hex}"


def _extract_zip_preserving_symlinks(zip_file: ZipFile, dest_dir: str) -> None:
    # ZipFile.extractall() writes symlink entries out as regular files
    # containing their link target as text, since it ignores the archive's
    # unix permission bits. Restore them as actual symlinks instead.
    for info in zip_file.infolist():
        is_symlink = stat.S_ISLNK(info.external_attr >> 16)
        if is_symlink:
            target_path = os.path.join(dest_dir, info.filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            os.symlink(zip_file.read(info).decode(), target_path)
        else:
            zip_file.extract(info, dest_dir)


def download_wklibs_sample_archive(name: str) -> UPath:
    """Downloads `{name}.zip` from the wklibs-samples bucket and extracts it
    into CACHE_DIR, once ever (subsequent calls, including from later test
    runs, reuse the cached extraction)."""
    dest_dir = CACHE_DIR / name
    if not dest_dir.exists():
        tmp_dir = _tmp_cache_path(name)
        tmp_dir.mkdir(parents=True)
        try:
            # Streamed to a NamedTemporaryFile instead of straight to the
            # destination since ZipFile needs random access to extract it.
            with NamedTemporaryFile(suffix=".zip") as archive_file:
                with httpx.stream(
                    "GET",
                    f"{WKLIBS_SAMPLES_BASE_URL}/{name}.zip",
                    follow_redirects=True,
                ) as response:
                    for chunk in response.iter_bytes():
                        archive_file.write(chunk)
                with ZipFile(archive_file, "r") as zip_file:
                    _extract_zip_preserving_symlinks(zip_file, str(tmp_dir))
            # os.replace is atomic (both paths are under CACHE_DIR, i.e. the
            # same filesystem), so a crash mid-download never leaves a
            # half-extracted dest_dir behind.
            os.replace(str(tmp_dir / name), str(dest_dir))
        finally:
            rmtree(tmp_dir)  # a no-op if already moved away above
    return dest_dir


def create_synthetic_multi_timepoint_ims(
    path: UPath, *, num_timepoints: int, num_channels: int, z: int, y: int, x: int
) -> None:
    """Writes a minimal HDF5 structure matching what ImsChunkedImages actually
    reads (DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data). This
    intentionally skips the DataSetInfo attributes that the full
    imaris_ims_file_reader library needs, since callers monkeypatch
    ims_chunked_images._read_ims_metadata_quietly instead of relying on a
    byte-perfect Imaris file."""
    with h5py.File(str(path), "w") as f:
        res0 = f.create_group("DataSet").create_group("ResolutionLevel 0")
        for t in range(num_timepoints):
            tp = res0.create_group(f"TimePoint {t}")
            for c in range(num_channels):
                ch = tp.create_group(f"Channel {c}")
                # encodes (t, c) into every voxel so tests can verify both
                # axes were read correctly, independent of x/y/z position
                data = np.full((z, y, x), t * 100 + c, dtype=np.uint16)
                ch.create_dataset("Data", data=data)

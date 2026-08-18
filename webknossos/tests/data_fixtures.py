"""Test data fixtures: downloaded (mainly from the `wklibs-samples` bucket on
static.webknossos.org) or synthetically generated.

Downloads are cached under CACHE_DIR so repeated local test runs reuse
what's already there instead of re-fetching it every time. The cache is
gitignored and never cleaned up automatically; delete it by hand to force a
re-download."""

import json
import os
import stat
import uuid
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZipFile

import h5py
import httpx
import numpy as np
import tensorstore as ts
from numpy.typing import DTypeLike
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
    path: UPath,
    *,
    num_timepoints: int,
    num_channels: int,
    z: int,
    y: int,
    x: int,
    dtype: DTypeLike = np.uint16,
) -> None:
    """Writes a minimal HDF5 structure matching what an Imaris reader expects
    (DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data). Intentionally
    skips the DataSetInfo attributes a full Imaris file would need.

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


def write_zarr_v2_array(path: UPath, data: np.ndarray) -> None:
    """Writes `data` as a plain Zarr v2 array (a single chunk, no compression)
    via tensorstore, for tests of readers that consume a bare `.zarray`."""
    array = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(data.shape),
                "chunks": list(data.shape),
                "dtype": data.dtype.str,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
            },
        },
        create=True,
        context=ts.Context(),
    ).result()
    array[...] = data


def write_zarr_v3_array(
    path: UPath, data: np.ndarray, *, dimension_names: list[str] | None = None
) -> None:
    """Writes `data` as a plain Zarr v3 array (a single chunk, no compression,
    no sharding) via tensorstore. `dimension_names` is Zarr v3's own axis-name
    metadata, independent of any OME wrapper — omit it to test the positional
    axis fallback instead."""
    metadata: dict[str, Any] = {
        "data_type": data.dtype.name,
        "shape": list(data.shape),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(data.shape)},
        },
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
    if dimension_names is not None:
        metadata["dimension_names"] = dimension_names
    array = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": metadata,
        },
        create=True,
        context=ts.Context(),
    ).result()
    array[...] = data


def write_n5_array(
    path: UPath, data: np.ndarray, *, downsampling_factors: list[int] | None = None
) -> None:
    """Writes `data` as a plain N5 dataset (a single block, no compression)
    via tensorstore. `downsampling_factors`, if given, is added to
    `attributes.json` afterwards, as a real N5 pyramid level would carry."""
    array = ts.open(
        {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "dimensions": list(data.shape),
                "blockSize": list(data.shape),
                "dataType": data.dtype.name,
                "compression": {"type": "raw"},
            },
        },
        create=True,
        context=ts.Context(),
    ).result()
    array[...] = data
    if downsampling_factors is not None:
        attributes_path = path / "attributes.json"
        attributes = json.loads(attributes_path.read_bytes())
        attributes["downsamplingFactors"] = downsampling_factors
        attributes_path.write_text(json.dumps(attributes))


def write_neuroglancer_precomputed_scale(
    path: UPath,
    data_xyzc: np.ndarray,
    *,
    resolution: tuple[float, float, float] = (4.0, 4.0, 4.0),
    key: str | None = None,
) -> None:
    """Writes one scale of a neuroglancer precomputed volume at `path`
    (its `info` file's root) via tensorstore. `data_xyzc` is in the format's
    own native axis order, (x, y, z, channel) — call again with a different
    `resolution`/`key` to add further scales to the same volume."""
    x, y, z, num_channels = data_xyzc.shape
    scale_key = key or "_".join(str(int(r)) for r in resolution)
    array = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(path)},
            "multiscale_metadata": {
                "type": "image",
                "data_type": data_xyzc.dtype.name,
                "num_channels": num_channels,
            },
            "scale_metadata": {
                "key": scale_key,
                "size": [x, y, z],
                "resolution": list(resolution),
                "chunk_size": [x, y, z],
                "encoding": "raw",
            },
        },
        create=True,
        context=ts.Context(),
    ).result()
    array[...] = data_xyzc


def write_ome_zarr_v2_group(
    path: UPath,
    datasets: list[tuple[str, np.ndarray, list[float]]],
    axes: list[dict[str, str]],
) -> None:
    """Writes a v2 (`.zgroup`/`.zattrs`) OME-Zarr (NGFF 0.4) multiscale group.
    Each entry in `datasets` is `(relative_path, data, scale)`: `data` is
    written as a plain v2 sub-array at `path/relative_path`, and `scale`
    becomes its `coordinateTransformations`."""
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (path / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": axes,
                        "datasets": [
                            {
                                "path": rel_path,
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": scale}
                                ],
                            }
                            for rel_path, _, scale in datasets
                        ],
                    }
                ]
            }
        )
    )
    for rel_path, data, _ in datasets:
        write_zarr_v2_array(path / rel_path, data)


def write_ome_zarr_v3_group(
    path: UPath,
    datasets: list[tuple[str, np.ndarray, list[float]]],
    axes: list[dict[str, str]],
) -> None:
    """Writes a v3 (`zarr.json`) OME-Zarr (NGFF 0.5) multiscale group. Each
    entry in `datasets` is `(relative_path, data, scale)`: `data` is written
    as a plain v3 sub-array at `path/relative_path`, and `scale` becomes its
    `coordinateTransformations`."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "version": "0.5",
                        "multiscales": [
                            {
                                "axes": axes,
                                "datasets": [
                                    {
                                        "path": rel_path,
                                        "coordinateTransformations": [
                                            {"type": "scale", "scale": scale}
                                        ],
                                    }
                                    for rel_path, _, scale in datasets
                                ],
                            }
                        ],
                    }
                },
            }
        )
    )
    for rel_path, data, _ in datasets:
        write_zarr_v3_array(path / rel_path, data)

import importlib.util
import json
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import tensorstore as ts
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

import importlib.util
import json
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import h5py
import httpx
import numpy as np
import pytest
import tensorstore as ts
from numpy.typing import DTypeLike
from upath import UPath

from webknossos.dataset._utils.tensorstore_helpers import _make_kvstore

_TS_CONTEXT = ts.Context()

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


def create_synthetic_zarr3_array(
    path: UPath, data: np.ndarray, *, dimension_names: Sequence[str] | None = None
) -> None:
    """Writes `data` as a real Zarr v3 array via tensorstore, one chunk. Set
    `dimension_names` to give it labeled axes (Zarr v3's `dimension_names`);
    omitted, the array carries no axis names, exercising positional guessing."""
    metadata: dict[str, Any] = {
        "data_type": str(data.dtype),
        "shape": list(data.shape),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(data.shape)},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
    }
    if dimension_names is not None:
        metadata["dimension_names"] = list(dimension_names)
    arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": _make_kvstore(path),
            "metadata": metadata,
        },
        create=True,
        context=_TS_CONTEXT,
    ).result()
    arr[:].write(data).result()


def create_synthetic_zarr2_array(path: UPath, data: np.ndarray) -> None:
    """Writes `data` as a real Zarr v2 array via tensorstore, one chunk. Zarr
    v2 has no `dimension_names` equivalent, so a bare array like this always
    exercises positional axis guessing."""
    arr = ts.open(
        {
            "driver": "zarr",
            "kvstore": _make_kvstore(path),
            "metadata": {
                "shape": list(data.shape),
                "chunks": list(data.shape),
                "dtype": data.dtype.str,
                "fill_value": 0,
                "order": "C",
                "compressor": None,
                "filters": None,
                "dimension_separator": "/",
            },
        },
        create=True,
        context=_TS_CONTEXT,
    ).result()
    arr[:].write(data).result()


def create_synthetic_ome_zarr_multiscale(
    path: UPath,
    levels: Sequence[np.ndarray],
    *,
    zarr_version: int,
    axes_names: Sequence[str] = ("z", "y", "x"),
    scales: Sequence[Sequence[float]] | None = None,
    dataset_paths: Sequence[str] | None = None,
) -> None:
    """Writes an OME-NGFF multiscale group with one array per level —
    NGFF 0.5 on Zarr v3 (`zarr_version=3`) or NGFF 0.4 on Zarr v2
    (`zarr_version=2`). `levels[0]` is the finest resolution; `scales`
    defaults to `2**index` per spatial axis, so passing `levels` out of
    finest-first order (with `dataset_paths` to match) still lets a test
    verify the reader picks the finest one by scale, not by list position.
    """
    if scales is None:
        scales = [[2.0**i] * len(axes_names) for i in range(len(levels))]
    if dataset_paths is None:
        dataset_paths = [str(i) for i in range(len(levels))]

    axes = [{"name": name, "type": "space"} for name in axes_names]
    datasets = [
        {
            "path": dataset_path,
            "coordinateTransformations": [{"type": "scale", "scale": list(scale)}],
        }
        for dataset_path, scale in zip(dataset_paths, scales, strict=True)
    ]

    if zarr_version == 3:
        for dataset_path, level in zip(dataset_paths, levels, strict=True):
            create_synthetic_zarr3_array(path / dataset_path, level)
        (path / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "ome": {
                            "version": "0.5",
                            "multiscales": [{"axes": axes, "datasets": datasets}],
                        }
                    },
                }
            )
        )
    elif zarr_version == 2:
        for dataset_path, level in zip(dataset_paths, levels, strict=True):
            create_synthetic_zarr2_array(path / dataset_path, level)
        (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (path / ".zattrs").write_text(
            json.dumps(
                {
                    "multiscales": [
                        {"version": "0.4", "axes": axes, "datasets": datasets}
                    ]
                }
            )
        )
    else:
        raise ValueError(f"Unsupported zarr_version {zarr_version}, expected 2 or 3.")


def create_synthetic_n5_dataset(path: UPath, data: np.ndarray) -> None:
    """Writes `data` as a real N5 dataset via tensorstore, one block. N5's
    attributes.json carries no axis names, so this always exercises
    positional axis guessing."""
    arr = ts.open(
        {
            "driver": "n5",
            "kvstore": _make_kvstore(path),
            "metadata": {
                "dimensions": list(data.shape),
                "blockSize": list(data.shape),
                "dataType": str(data.dtype),
                "compression": {"type": "raw"},
            },
        },
        create=True,
        context=_TS_CONTEXT,
    ).result()
    arr[:].write(data).result()


def create_synthetic_n5_pyramid(
    path: UPath,
    levels: Sequence[np.ndarray],
    *,
    level_names: Sequence[str] | None = None,
    downsampling_factors: Sequence[Sequence[int]] | None = None,
) -> None:
    """Writes an N5 multiscale pyramid group: one dataset per level, named
    `s0`/`s1`/... by default (`levels[0]` = finest = `s0`), each carrying its
    own `downsamplingFactors` attribute (`2**index` per axis by default)."""
    if level_names is None:
        level_names = [f"s{i}" for i in range(len(levels))]
    if downsampling_factors is None:
        downsampling_factors = [[2**i] * levels[0].ndim for i in range(len(levels))]
    for name, level, factors in zip(
        level_names, levels, downsampling_factors, strict=True
    ):
        level_path = path / name
        create_synthetic_n5_dataset(level_path, level)
        attrs_path = level_path / "attributes.json"
        attrs = json.loads(attrs_path.read_bytes())
        attrs["downsamplingFactors"] = list(factors)
        attrs_path.write_text(json.dumps(attrs))


def create_synthetic_neuroglancer_precomputed(
    path: UPath, scales: Sequence[tuple[Sequence[int], np.ndarray]]
) -> None:
    """Writes a real Neuroglancer precomputed volume via tensorstore, one
    chunk per scale. `scales` is a sequence of (resolution_xyz, data), where
    `data` has axis order (x, y, z, channel) — the format's own native order.
    Keys are derived from the resolution (`f"{x}_{y}_{z}"`), like real
    precomputed volumes; pass scales out of finest-first order to verify the
    reader picks the finest one by resolution, not by list position."""
    for resolution, data in scales:
        x, y, z, num_channels = data.shape
        key = "_".join(str(r) for r in resolution)
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": _make_kvstore(path),
            "multiscale_metadata": {
                "data_type": str(data.dtype),
                "num_channels": num_channels,
                "type": "image",
            },
            "scale_metadata": {
                "key": key,
                "resolution": list(resolution),
                "size": [x, y, z],
                "voxel_offset": [0, 0, 0],
                "chunk_size": [x, y, z],
                "encoding": "raw",
            },
        }
        arr = ts.open(spec, create=True, context=_TS_CONTEXT).result()
        arr[:].write(data).result()

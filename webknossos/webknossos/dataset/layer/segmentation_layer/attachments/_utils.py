import math
from typing import Any

import numpy as np
from upath import UPath


def read_zarr3_array(path: UPath) -> np.ndarray:  # type: ignore[type-arg]
    """Read a Zarr v3 array from disk into a numpy array."""
    import tensorstore as ts

    from webknossos.dataset.layer.view._array import TS_CONTEXT

    arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
        },
        open=True,
        context=TS_CONTEXT,
    ).result()
    return arr[:].read().result()  # type: ignore[no-any-return]


def write_zarr3_array(
    path: UPath,
    data: np.ndarray,  # type: ignore[type-arg]
    *,
    dtype: np.typing.DTypeLike,
    target_chunk_size_bytes: int,
    target_shard_size_bytes: int,
) -> None:
    """Write a numpy array as a Zarr v3 sharded array using tensorstore.

    Chunk and shard shapes are derived from the array's shape and dtype so
    that each chunk approximates `target_chunk_size_bytes` and each shard
    approximates `target_shard_size_bytes`.  The first axis is used as the
    "row" axis; remaining axes are kept whole in every chunk/shard.
    The shard shape is always a multiple of the chunk shape.
    """
    import tensorstore as ts

    from webknossos.dataset.layer.view._array import TS_CONTEXT

    np_dtype = np.dtype(dtype)
    # bytes consumed by one step along axis 0
    row_bytes = (
        np_dtype.itemsize * math.prod(data.shape[1:])
        if data.ndim > 1
        else np_dtype.itemsize
    )

    n_rows = data.shape[0] if data.shape[0] > 0 else 1
    chunk_rows = max(1, min(n_rows, target_chunk_size_bytes // row_bytes))
    shard_cap = max(chunk_rows, min(n_rows, target_shard_size_bytes // row_bytes))
    # round up to the nearest multiple of chunk_rows
    shard_rows = ((shard_cap + chunk_rows - 1) // chunk_rows) * chunk_rows

    chunk_shape = (chunk_rows, *data.shape[1:])
    shard_shape = (shard_rows, *data.shape[1:])

    codecs: list[dict[str, Any]] = [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": list(chunk_shape),
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "zstd", "configuration": {"level": 5, "checksum": True}},
                ],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ],
            },
        }
    ]

    metadata: dict[str, Any] = {
        "data_type": np_dtype.name,
        "shape": list(data.shape),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(shard_shape)},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": codecs,
    }
    arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": metadata,
        },
        create=True,
        context=TS_CONTEXT,
    ).result()
    arr[:].write(data).result()

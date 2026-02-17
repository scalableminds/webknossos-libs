import os

import numpy as np

from ..dataset_properties.data_format import DataFormat
from ..geometry import Vec3Int

DEFAULT_CHUNK_SHAPE = Vec3Int.full(32)
DEFAULT_DATA_FORMAT = (
    DataFormat(os.environ["WK_DEFAULT_DATA_FORMAT"])
    if "WK_DEFAULT_DATA_FORMAT" in os.environ
    else DataFormat.Zarr3
)

DEFAULT_CHUNKS_PER_SHARD = (
    Vec3Int.from_str(os.environ["WK_DEFAULT_CHUNKS_PER_SHARD"])
    if "WK_DEFAULT_CHUNKS_PER_SHARD" in os.environ
    else Vec3Int.full(32)
)
DEFAULT_CHUNKS_PER_SHARD_ZARR = Vec3Int.full(1)
DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES = Vec3Int(128, 128, 1)
DEFAULT_SHARD_SHAPE = DEFAULT_CHUNKS_PER_SHARD * DEFAULT_CHUNK_SHAPE
DEFAULT_SHARD_SHAPE_FROM_IMAGES = (
    DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES * DEFAULT_CHUNK_SHAPE
)

DEFAULT_DTYPE = np.dtype("uint8")
DEFAULT_BIT_DEPTH = 8
PROPERTIES_FILE_NAME = "datasource-properties.json"
ZGROUP_FILE_NAME = ".zgroup"
ZATTRS_FILE_NAME = ".zattrs"
ZARR_JSON_FILE_NAME = "zarr.json"

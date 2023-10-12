from ..geometry import Vec3Int
from .data_format import DataFormat

DEFAULT_WKW_FILE_LEN = 32
DEFAULT_CHUNK_SHAPE = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD_ZARR = Vec3Int.full(1)
DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES = Vec3Int(128, 128, 1)
DEFAULT_BIT_DEPTH = 8
DEFAULT_DATA_FORMAT = DataFormat.WKW
PROPERTIES_FILE_NAME = "datasource-properties.json"
ZGROUP_FILE_NAME = ".zgroup"
ZATTRS_FILE_NAME = ".zattrs"
ZARR_JSON_FILE_NAME = "zarr.json"

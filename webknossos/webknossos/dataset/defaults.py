from os import environ

from webknossos.geometry import Vec3Int

DEFAULT_WKW_FILE_LEN = 32
DEFAULT_CHUNK_SHAPE = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD_ZARR = (
    Vec3Int.full(32)
    if environ.get("WK_USE_ZARRITA", None) is not None
    else Vec3Int.full(1)
)

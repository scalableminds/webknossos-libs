from os import environ

from webknossos.geometry import Vec3Int

WK_USE_ZARRITA = environ.get("WK_USE_ZARRITA") is not None
DEFAULT_WKW_FILE_LEN = 32
DEFAULT_CHUNK_SHAPE = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD = Vec3Int.full(32)
DEFAULT_CHUNKS_PER_SHARD_ZARR = Vec3Int.full(1)
DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES = Vec3Int(128, 128, 1)

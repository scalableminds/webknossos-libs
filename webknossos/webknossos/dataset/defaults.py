import copyreg
import os
import ssl
from collections.abc import Callable
from typing import Any

import certifi

from ..geometry import Vec3Int
from .data_format import DataFormat


def _create_sslcontext() -> ssl.SSLContext:
    cafile = certifi.where()
    ssl_context = ssl.create_default_context(cafile=cafile)
    ssl_context.cafile = cafile  # type: ignore
    return ssl_context


def _save_sslcontext(
    obj: ssl.SSLContext,
) -> tuple[Callable[[Any, Any], ssl.SSLContext], tuple[ssl._SSLMethod, str | None]]:
    cafile = getattr(obj, "cafile", None)
    return _rebuild_sslcontext, (obj.protocol, cafile)


def _rebuild_sslcontext(protocol: ssl._SSLMethod, cafile: str | None) -> ssl.SSLContext:
    ssl_context = ssl.SSLContext(protocol)
    if cafile is not None:
        ssl_context.load_verify_locations(cafile=cafile)
        ssl_context.cafile = cafile  # type: ignore
    return ssl_context


copyreg.pickle(ssl.SSLContext, _save_sslcontext)

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

DEFAULT_BIT_DEPTH = 8
PROPERTIES_FILE_NAME = "datasource-properties.json"
ZGROUP_FILE_NAME = ".zgroup"
ZATTRS_FILE_NAME = ".zattrs"
ZARR_JSON_FILE_NAME = "zarr.json"
SSL_CONTEXT = _create_sslcontext()

DEFAULT_NUM_RETRIES = (
    int(os.environ["DEFAULT_NUM_RETRIES"])
    if "DEFAULT_NUM_RETRIES" in os.environ
    else 20
)
DEFAULT_BACKOFF_FACTOR = (
    float(os.environ["DEFAULT_BACKOFF_FACTOR"])
    if "DEFAULT_BACKOFF_FACTOR" in os.environ
    else 1.75
)

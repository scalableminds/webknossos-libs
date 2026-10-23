import atexit
import math
from functools import lru_cache
from hashlib import sha256
from tempfile import mkdtemp
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import tensorstore as ts
from upath import UPath

from ...utils import is_fs_path, rmtree

TS_CONTEXT = ts.Context()


class AWSCredentialManager:
    """Writes one credentials file per key pair, for tensorstore's `aws_credentials`.

    A file is written once and never changed afterwards. Readers therefore never see
    a partially written or rewritten file, without any locking between threads or
    forked processes that share the folder.
    """

    folder_path: UPath

    def __init__(self, folder_path: UPath) -> None:
        self.folder_path = folder_path
        # Profiles whose file is known to exist. The folder starts out empty, so
        # this replaces a disk check on each add. Without a lock, two threads may
        # both write the same new file, which is harmless.
        self._written_profiles: set[str] = set()
        self.config_file_path.write_text("[default]\n")

        atexit.register(self._cleanup)

    @property
    def config_file_path(self) -> UPath:
        return self.folder_path / "config"

    def credentials_file_path(self, profile: str) -> UPath:
        return self.folder_path / f"credentials-{profile}"

    def _write_credentials_file(
        self, path: UPath, profile: str, access_key_id: str, secret_access_key: str
    ) -> None:
        # Written to a unique temporary file and renamed into place, so that
        # tensorstore never reads a partially written file
        tmp_path = self.folder_path / f"{path.name}.{uuid4().hex}.tmp"
        try:
            tmp_path.write_text(
                f"[{profile}]\naws_access_key_id = {access_key_id}\naws_secret_access_key = {secret_access_key}\n"
            )
            try:
                tmp_path.rename(path)
            except FileExistsError:
                # On Windows, rename does not replace an existing file. Another
                # thread or process already wrote the same content.
                pass
        finally:
            tmp_path.unlink(missing_ok=True)

    def _cleanup(self) -> None:
        rmtree(self.folder_path)

    def add(self, access_key_id: str, secret_access_key: str) -> dict[str, str]:
        # A stable digest, so that equal key pairs map to the same file in all
        # processes. The secret is not recoverable from it.
        digest = sha256(f"{access_key_id}\0{secret_access_key}".encode()).hexdigest()
        profile = f"profile-{digest}"
        credentials_file_path = self.credentials_file_path(profile)
        if profile not in self._written_profiles:
            self._write_credentials_file(
                credentials_file_path, profile, access_key_id, secret_access_key
            )
            self._written_profiles.add(profile)
        return {
            "type": "profile",
            "profile": profile,
            "config_file": str(self.config_file_path),
            "credentials_file": str(credentials_file_path),
        }


@lru_cache
def _aws_credential_folder() -> UPath:
    return UPath(mkdtemp())


_aws_credential_manager = AWSCredentialManager(_aws_credential_folder())


def _make_kvstore(path: UPath) -> str | dict[str, str | list[str]]:
    if is_fs_path(path):
        return {"driver": "file", "path": str(path)}
    elif path.protocol in ("http", "https"):
        return {
            "driver": "http",
            "base_url": str(path),
            "headers": [
                f"{key}: {value}"
                for key, value in path.storage_options.get("headers", {}).items()
            ],
        }
    elif path.protocol in ("s3"):
        parsed_url = urlparse(str(path))
        kvstore_spec: dict[str, Any] = {
            "driver": "s3",
            "path": parsed_url.path.lstrip("/"),
            "bucket": parsed_url.netloc,
            "use_conditional_write": False,
        }
        if endpoint_url := path.storage_options.get("endpoint_url", None):
            kvstore_spec["endpoint"] = endpoint_url
        if "key" in path.storage_options and "secret" in path.storage_options:
            kvstore_spec["aws_credentials"] = _aws_credential_manager.add(
                path.storage_options["key"], path.storage_options["secret"]
            )
        else:
            kvstore_spec["aws_credentials"] = {"type": "default"}
        return kvstore_spec
    elif path.protocol in ("gs", "gcs"):
        parsed_url = urlparse(str(path))
        return {
            "driver": "gcs",
            "path": parsed_url.path.lstrip("/"),
            "bucket": parsed_url.netloc,
        }
    elif path.protocol == "memory":
        # use memory driver (in-memory file systems), e.g. useful for testing
        # attention: this is not a persistent storage and it does not support
        # multiprocessing since memory is not shared between processes
        return {
            "driver": "memory",
            "path": path.path,
        }
    else:
        return {
            "driver": "file",
            "path": str(path),
        }


def open_zarr3_array(path: UPath, context: ts.Context = TS_CONTEXT) -> "ts.TensorStore":
    """Open a Zarr v3 array from disk into a tensorstore array."""

    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": _make_kvstore(path),
        },
        open=True,
        context=context,
    ).result()


def read_zarr3_array(path: UPath) -> np.ndarray:
    """Read a Zarr v3 array from disk into a numpy array."""

    arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": _make_kvstore(path),
        },
        open=True,
        context=TS_CONTEXT,
    ).result()
    return arr[:].read().result()


def write_zarr3_array(
    path: UPath,
    data: np.ndarray,
    *,
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
    np_dtype = data.dtype
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
            "kvstore": _make_kvstore(path),
            "metadata": metadata,
        },
        create=True,
        context=TS_CONTEXT,
    ).result()
    arr[:].write(data).result()

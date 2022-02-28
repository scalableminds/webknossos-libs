import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import wkw

from webknossos.geometry import Vec3Int


@dataclass
class StorageBackendInfo:
    num_channels: int
    voxel_type: np.dtype
    chunk_size: Vec3Int
    chunks_per_shard: Vec3Int
    compression_mode: bool = False

    @property
    def shard_size(self) -> Vec3Int:
        return self.chunk_size * self.chunks_per_shard


class WKWStorageBackend:
    _path: Path
    _cached_wkw_dataset: Optional[wkw.Dataset]

    def __init__(self, path: Path):
        self._path = path
        self._cached_backend = None

    def compress_shard(self, source_path: Path, target_path: Path) -> None:
        wkw.File.compress(str(source_path), str(target_path))

    def compress(self, target_path: Path) -> None:
        with wkw.Dataset.open(str(self._path)) as wkw_dataset:
            wkw_dataset.compress(str(target_path))

    @property
    def info(self) -> StorageBackendInfo:
        with wkw.Dataset.open(str(self._path)) as wkw_dataset:
            return StorageBackendInfo(
                num_channels=wkw_dataset.header.num_channels,
                voxel_type=wkw_dataset.header.voxel_type,
                compression_mode=wkw_dataset.header.block_type
                != wkw.Header.BLOCK_TYPE_RAW,
                chunk_size=Vec3Int.full(wkw_dataset.header.block_len),
                chunks_per_shard=Vec3Int.full(
                    wkw_dataset.header.file_len,
                ),
            )

    @classmethod
    def create(
        cls, path: Path, storage_info: StorageBackendInfo
    ) -> "WKWStorageBackend":
        assert storage_info.chunk_size[0] == storage_info.chunk_size[1]
        assert storage_info.chunk_size[0] == storage_info.chunk_size[2]
        assert storage_info.chunks_per_shard[0] == storage_info.chunks_per_shard[1]
        assert storage_info.chunks_per_shard[0] == storage_info.chunks_per_shard[2]
        wkw.Dataset.create(
            str(path),
            wkw.Header(
                voxel_type=storage_info.voxel_type,
                num_channels=storage_info.num_channels,
                block_len=storage_info.chunk_size[0],
                file_len=storage_info.chunks_per_shard[0],
                block_type=(
                    wkw.Header.BLOCK_TYPE_LZ4HC
                    if storage_info.compression_mode
                    else wkw.Header.BLOCK_TYPE_RAW
                ),
            ),
        ).close()
        return WKWStorageBackend(path)

    def remove(self) -> None:
        self.close()
        shutil.rmtree(self._path)

    def move(self, target_path: Path) -> "WKWStorageBackend":
        self.close()
        shutil.move(str(self._path), target_path)
        return WKWStorageBackend(target_path)

    def read(self, offset, shape) -> np.ndarray:
        return self._wkw_dataset.read(offset, shape)

    def write(self, offset, data) -> None:
        self._wkw_dataset.write(offset, data)

    def list_files(self) -> List[Path]:
        return [Path(path) for path in self._wkw_dataset.list_files()]

    def close(self) -> None:
        if self._cached_wkw_dataset is not None:
            self._cached_wkw_dataset.close()
            self._cached_wkw_dataset = None

    @property
    def _wkw_dataset(self) -> wkw.Dataset:
        if self._cached_wkw_dataset is None:
            self._cached_wkw_dataset = wkw.Dataset.open(
                str(self._path)
            )  # No need to pass the header to the wkw.Dataset
        return self._cached_wkw_dataset

    @_wkw_dataset.deleter
    def _wkw_dataset(self) -> None:
        self.close()

    def __del__(self) -> None:
        del self._cached_wkw_dataset

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_wkw_dataset"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_wkw_dataset"] = None
        self.__dict__ = d

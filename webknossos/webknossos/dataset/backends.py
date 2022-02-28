import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import wkw
import zarr
from numcodecs import Blosc

from webknossos.geometry import Vec3Int, Vec3IntLike


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


class StorageBackend(ABC):
    @property
    @abstractmethod
    def info(self) -> StorageBackendInfo:
        pass

    @abstractmethod
    def compress_shard(self, source_path: Path, target_path: Path) -> None:
        pass

    @abstractmethod
    def compress(self, target_path: Path) -> None:
        pass

    @abstractmethod
    def remove(self) -> None:
        pass

    @abstractmethod
    def move(self, target_path: Path) -> "StorageBackend":
        pass

    @abstractmethod
    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def list_files(self) -> List[Path]:
        pass


class WKWStorageBackend(StorageBackend):
    _path: Path
    _cached_wkw_dataset: Optional[wkw.Dataset]

    def __init__(self, path: Path):
        self._path = path
        self._cached_wkw_dataset = None

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
        self._close()
        shutil.rmtree(self._path)

    def move(self, target_path: Path) -> "WKWStorageBackend":
        self._close()
        shutil.move(str(self._path), target_path)
        return WKWStorageBackend(target_path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        return self._wkw_dataset.read(offset, shape)

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        self._wkw_dataset.write(offset, data)

    def list_files(self) -> List[Path]:
        return [Path(path) for path in self._wkw_dataset.list_files()]

    def _close(self) -> None:
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
        self._close()

    def __del__(self) -> None:
        del self._cached_wkw_dataset

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_wkw_dataset"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_wkw_dataset"] = None
        self.__dict__ = d


class ZarrStorageBackend(StorageBackend):
    _path: Path

    def __init__(self, path: Path):
        self._path = path

    def compress_shard(self, source_path: Path, target_path: Path) -> None:
        raise NotImplementedError()

    def compress(self, target_path: Path) -> None:
        raise NotImplementedError()

    @property
    def info(self) -> StorageBackendInfo:
        zarray = zarr.open_array(self._path)
        return StorageBackendInfo(
            num_channels=zarray.shape[0],
            voxel_type=zarray.dtype,
            compression_mode=zarray.compressor is not None,
            chunk_size=zarray.chunks or Vec3Int.full(1),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(
        cls, path: Path, storage_info: StorageBackendInfo
    ) -> "ZarrStorageBackend":
        assert storage_info.chunks_per_shard == Vec3Int.full(1)
        zarr.create(
            (storage_info.num_channels, 0, 0, 0),
            chunks=storage_info.chunk_size,
            dtype=storage_info.voxel_type,
            compressor=(
                Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
                if storage_info.compression_mode
                else None
            ),
            store=path,
        )
        return ZarrStorageBackend(path)

    def remove(self) -> None:
        shutil.rmtree(self._path)

    def move(self, target_path: Path) -> "ZarrStorageBackend":
        shutil.move(str(self._path), target_path)
        return ZarrStorageBackend(target_path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        offset = Vec3Int(offset)
        shape = Vec3Int(shape)
        return zarr.open(store=self._path)[
            :,
            offset.x : (offset.x + shape.x),
            offset.y : (offset.y + shape.y),
            offset.z : (offset.z + shape.z),
        ]

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        offset = Vec3Int(offset)
        zarray = zarr.open(store=self._path)
        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4
        zarray[
            :,
            offset.x : (offset.x + data.shape[1]),
            offset.y : (offset.y + data.shape[2]),
            offset.z : (offset.z + data.shape[3]),
        ] = data

    def list_files(self) -> List[Path]:
        raise NotImplementedError()

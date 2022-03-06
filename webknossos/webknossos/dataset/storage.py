import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Type

import numpy as np
import wkw
import zarr
from numcodecs import Blosc

from webknossos.geometry import BoundingBox, Vec3Int, Vec3IntLike


def is_power_of_two(num: int) -> bool:
    return num & (num - 1) == 0


class StorageArrayException(Exception):
    pass


@dataclass
class StorageArrayInfo:
    data_format: str
    num_channels: int
    voxel_type: np.dtype
    chunk_size: Vec3Int
    chunks_per_shard: Vec3Int
    compression_mode: bool = False

    @property
    def shard_size(self) -> Vec3Int:
        return self.chunk_size * self.chunks_per_shard


class StorageArray(ABC):
    data_format = ""

    _path: Path

    def __init__(self, path: Path):
        self._path = path

    @property
    @abstractmethod
    def info(self) -> StorageArrayInfo:
        pass

    @classmethod
    @abstractmethod
    def try_open(_cls, path: Path) -> Optional["StorageArray"]:
        for cls in (WKWStorageArray, ZarrStorageArray):
            array_maybe = cls.try_open(path)
            if array_maybe is not None:
                return array_maybe
        return None

    @classmethod
    @abstractmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "StorageArray":
        pass

    @abstractmethod
    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def resize(
        self, new_shape: Vec3IntLike, align_with_shards: bool = True, warn: bool = False
    ) -> None:
        pass

    @abstractmethod
    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def get_class(data_format: str) -> Type["StorageArray"]:
        for cls in (WKWStorageArray, ZarrStorageArray):
            if cls.data_format == data_format:
                return cls
        raise ValueError(f"Data format `{data_format}` is invalid.")


class WKWStorageArray(StorageArray):
    data_format = "wkw"

    _cached_wkw_dataset: Optional[wkw.Dataset]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_wkw_dataset = None

    @classmethod
    def try_open(cls, path: Path) -> Optional["WKWStorageArray"]:
        if (path / "header.wkw").is_file():
            return cls(path)
        return None

    @property
    def info(self) -> StorageArrayInfo:
        try:
            with wkw.Dataset.open(str(self._path)) as wkw_dataset:
                return StorageArrayInfo(
                    data_format=self.data_format,
                    num_channels=wkw_dataset.header.num_channels,
                    voxel_type=wkw_dataset.header.voxel_type,
                    compression_mode=wkw_dataset.header.block_type
                    != wkw.Header.BLOCK_TYPE_RAW,
                    chunk_size=Vec3Int.full(wkw_dataset.header.block_len),
                    chunks_per_shard=Vec3Int.full(
                        wkw_dataset.header.file_len,
                    ),
                )
        except wkw.wkw.WKWException as e:
            raise StorageArrayException(
                f"Exception while fetching storage info for {self._path}"
            ) from e

    @classmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "WKWStorageArray":
        assert storage_info.data_format == cls.data_format

        assert (
            storage_info.chunk_size.is_uniform()
        ), f"`chunk_size` needs to be uniform for WKW storage. Got {storage_info.chunk_size}."
        assert is_power_of_two(
            storage_info.chunk_size.x
        ), f"`chunk_size` needs to be a power of 2 for WKW storage. Got {storage_info.chunk_size.x}."
        assert (
            1 <= storage_info.chunk_size.x and storage_info.chunk_size.x <= 32768
        ), f"`chunk_size` needs to be a value between 1 and 32768 for WKW storage. Got {storage_info.chunk_size.x}."

        assert (
            storage_info.chunks_per_shard.is_uniform()
        ), f"`chunks_per_shard` needs to be uniform for WKW storage. Got {storage_info.chunks_per_shard}."
        assert is_power_of_two(
            storage_info.chunks_per_shard.x
        ), f"`chunks_per_shard` needs to be a power of 2 for WKW storage. Got {storage_info.chunks_per_shard.x}."
        assert (
            1 <= storage_info.chunks_per_shard.x
            and storage_info.chunks_per_shard.x <= 32768
        ), f"`chunks_per_shard` needs to be a value between 1 and 32768 for WKW storage. Got {storage_info.chunks_per_shard.x}."

        try:
            wkw.Dataset.create(
                str(path),
                wkw.Header(
                    voxel_type=storage_info.voxel_type,
                    num_channels=storage_info.num_channels,
                    block_len=storage_info.chunk_size.x,
                    file_len=storage_info.chunks_per_shard.x,
                    block_type=(
                        wkw.Header.BLOCK_TYPE_LZ4HC
                        if storage_info.compression_mode
                        else wkw.Header.BLOCK_TYPE_RAW
                    ),
                ),
            ).close()
        except wkw.wkw.WKWException as e:
            raise StorageArrayException(f"Exception while creating array {path}") from e
        return WKWStorageArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        return self._wkw_dataset.read(
            Vec3Int(offset).to_tuple(), Vec3Int(shape).to_tuple()
        )

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        self._wkw_dataset.write(Vec3Int(offset).to_tuple(), data)

    def resize(
        self,
        new_shape: Vec3IntLike,
        align_with_shards: bool = True,
        warn: bool = False,
    ) -> None:
        pass

    def _list_files(self) -> Iterator[Path]:
        return (
            Path(relpath(filename, self._path))
            for filename in self._wkw_dataset.list_files()
        )

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        def _extract_num(s: str) -> int:
            match = re.search("[0-9]+", s)
            assert match is not None
            return int(match[0])

        def _extract_file_index(file_path: Path) -> Vec3Int:
            zyx_index = [_extract_num(el) for el in file_path.parts]
            return Vec3Int(zyx_index[2], zyx_index[1], zyx_index[0])

        shard_size = self.info.shard_size
        for file_path in self._list_files():
            cube_index = _extract_file_index(file_path)
            cube_offset = cube_index * shard_size

            yield BoundingBox(cube_offset, shard_size)

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


class ZarrStorageArray(StorageArray):
    data_format = "zarr"

    @classmethod
    def try_open(cls, path: Path) -> Optional["ZarrStorageArray"]:
        if (path / ".zarray").is_file():
            return cls(path)
        return None

    @property
    def info(self) -> StorageArrayInfo:
        zarray = zarr.open_array(self._path, mode="r")
        return StorageArrayInfo(
            data_format=self.data_format,
            num_channels=zarray.shape[0],
            voxel_type=zarray.dtype,
            compression_mode=zarray.compressor is not None,
            chunk_size=Vec3Int(*zarray.chunks[1:4]) or Vec3Int.full(1),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(cls, path: Path, storage_info: StorageArrayInfo) -> "ZarrStorageArray":
        assert storage_info.data_format == cls.data_format
        assert storage_info.chunks_per_shard == Vec3Int.full(
            1
        ), "Zarr storage doesn't support sharding yet"
        zarr.create(
            shape=(storage_info.num_channels, 1, 1, 1),
            chunks=(storage_info.num_channels,) + storage_info.chunk_size.to_tuple(),
            dtype=storage_info.voxel_type,
            compressor=(
                Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
                if storage_info.compression_mode
                else None
            ),
            store=path,
        )
        return ZarrStorageArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        offset = Vec3Int(offset)
        shape = Vec3Int(shape)
        zarray = zarr.open_array(store=self._path, mode="r")
        data = zarray[
            :,
            offset.x : (offset.x + shape.x),
            offset.y : (offset.y + shape.y),
            offset.z : (offset.z + shape.z),
        ]
        if data.shape != shape:
            padded_data = np.zeros(
                (self.info.num_channels,) + shape.to_tuple(), dtype=data.dtype
            )
            padded_data[
                :,
                0 : data.shape[1],
                0 : data.shape[2],
                0 : data.shape[3],
            ] = data
            data = padded_data
        return data

    def resize(
        self, new_shape: Vec3IntLike, align_with_shards: bool = True, warn: bool = False
    ) -> None:
        new_shape = Vec3Int(new_shape)
        zarray = zarr.open_array(store=self._path, mode="a")

        new_shape_tuple = (
            zarray.shape[0],
            max(zarray.shape[1], new_shape.x),
            max(zarray.shape[2], new_shape.y),
            max(zarray.shape[3], new_shape.z),
        )
        if new_shape_tuple != zarray.shape:

            if align_with_shards:
                chunk_size = Vec3Int(zarray.chunks[1:4])
                chunks_per_shard = Vec3Int.full(1)
                shard_size = chunk_size * chunks_per_shard
                new_shape = new_shape.ceildiv(shard_size) * shard_size
                new_shape_tuple = (zarray.shape[0],) + new_shape.to_tuple()

            if warn:
                warnings.warn(
                    f"[WARNING] Resizing zarr storage array from `{zarray.shape}` to `{new_shape_tuple}`."
                )
            zarray.resize(new_shape_tuple)

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        offset = Vec3Int(offset)

        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4

        self.resize(offset + Vec3Int(data.shape[1:4]), warn=True)
        zarray = zarr.open_array(store=self._path, mode="a")
        zarray[
            :,
            offset.x : (offset.x + data.shape[1]),
            offset.y : (offset.y + data.shape[2]),
            offset.z : (offset.z + data.shape[3]),
        ] = data

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        zarray = zarr.open_array(store=self._path, mode="r")
        chunk_size = Vec3Int(*zarray.chunks[1:4])
        for key in zarray.store.keys():
            if not key.startswith("."):
                key_parts = [int(p) for p in key.split(zarray._dimension_separator)]
                chunk_idx = Vec3Int(key_parts[1:4])
                yield BoundingBox(topleft=chunk_idx * chunk_size, size=chunk_size)

    def close(self) -> None:
        pass

import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type, Union

import numcodecs
import numpy as np
import tensorstore as ts
import wkw
import zarr
from upath import UPath
from zarr.storage import FSStore

from ..geometry import BoundingBox, Vec3Int, Vec3IntLike
from ..utils import warn_deprecated
from .data_format import DataFormat


def _is_power_of_two(num: int) -> bool:
    return num & (num - 1) == 0


def _fsstore_from_path(path: Path, mode: str = "a") -> FSStore:
    storage_options: Dict[str, Any] = {}
    if isinstance(path, UPath):
        storage_options = getattr(path, "_kwargs", {}).copy()
        storage_options.pop("_url", None)
        return FSStore(url=str(path), mode=mode, **storage_options)

    return FSStore(url=str(path), mode=mode, **storage_options)


@contextmanager
def _blosc_disable_threading() -> Iterator[None]:
    old_value = numcodecs.blosc.use_threads

    # See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
    numcodecs.blosc.use_threads = False
    try:
        yield
    finally:
        numcodecs.blosc.use_threads = old_value


class ArrayException(Exception):
    pass


@dataclass
class ArrayInfo:
    data_format: DataFormat
    num_channels: int
    voxel_type: np.dtype
    chunk_shape: Vec3Int
    chunks_per_shard: Vec3Int
    compression_mode: bool = False

    @property
    def shard_size(self) -> Vec3Int:
        warn_deprecated("shard_size", "shard_shape")
        return self.shard_shape

    @property
    def shard_shape(self) -> Vec3Int:
        return self.chunk_shape * self.chunks_per_shard


class BaseArray(ABC):
    data_format = DataFormat.WKW

    _path: Path

    def __init__(self, path: Path):
        self._path = path

    @property
    @abstractmethod
    def info(self) -> ArrayInfo:
        pass

    @classmethod
    @abstractmethod
    def open(_cls, path: Path) -> "BaseArray":
        classes = (WKWArray, TensorStoreArray, ZarrArray)
        for cls in classes:
            try:
                array = cls.open(path)
                return array
            except ArrayException as e:
                pass
        raise ArrayException(f"Could not open the array at {path}.")

    @classmethod
    @abstractmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "BaseArray":
        pass

    @abstractmethod
    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def ensure_size(
        self, new_shape: Vec3IntLike, align_with_shards: bool = True, warn: bool = False
    ) -> None:
        pass

    @abstractmethod
    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        "The bounding boxes are measured in voxels of the current mag."

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def get_class(data_format: DataFormat) -> Type["BaseArray"]:
        if data_format == DataFormat.WKW:
            return WKWArray
        if DataFormat.Zarr3:
            return TensorStoreArray
        if data_format == DataFormat.Zarr:
            return ZarrArray
        raise ValueError(f"Array format `{data_format}` is invalid.")


class WKWArray(BaseArray):
    data_format = DataFormat.WKW

    _cached_wkw_dataset: Optional[wkw.Dataset]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_wkw_dataset = None

    @classmethod
    def open(cls, path: Path) -> "WKWArray":
        header_path = path / "header.wkw"

        if header_path.exists() and header_path.is_file():
            return cls(path)
        raise ArrayException(
            f"Could not open WKW array at {path}. `header.wkw` not found."
        )

    @property
    def info(self) -> ArrayInfo:
        header = self._wkw_dataset.header
        return ArrayInfo(
            data_format=self.data_format,
            num_channels=header.num_channels,
            voxel_type=header.voxel_type,
            compression_mode=header.block_type != wkw.Header.BLOCK_TYPE_RAW,
            chunk_shape=Vec3Int.full(header.block_len),
            chunks_per_shard=Vec3Int.full(
                header.file_len,
            ),
        )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "WKWArray":
        assert array_info.data_format == cls.data_format

        assert (
            array_info.chunk_shape.is_uniform()
        ), f"`chunk_shape` needs to be uniform for WKW storage. Got {array_info.chunk_shape}."
        assert _is_power_of_two(
            array_info.chunk_shape.x
        ), f"`chunk_shape` needs to be a power of 2 for WKW storage. Got {array_info.chunk_shape.x}."
        assert (
            1 <= array_info.chunk_shape.x and array_info.chunk_shape.x <= 32768
        ), f"`chunk_shape` needs to be a value between 1 and 32768 for WKW storage. Got {array_info.chunk_shape.x}."

        assert (
            array_info.chunks_per_shard.is_uniform()
        ), f"`chunks_per_shard` needs to be uniform for WKW storage. Got {array_info.chunks_per_shard}."
        assert _is_power_of_two(
            array_info.chunks_per_shard.x
        ), f"`chunks_per_shard` needs to be a power of 2 for WKW storage. Got {array_info.chunks_per_shard.x}."
        assert (
            1 <= array_info.chunks_per_shard.x
            and array_info.chunks_per_shard.x <= 32768
        ), f"`chunks_per_shard` needs to be a value between 1 and 32768 for WKW storage. Got {array_info.chunks_per_shard.x}."

        try:
            wkw.Dataset.create(
                str(path),
                wkw.Header(
                    voxel_type=array_info.voxel_type,
                    num_channels=array_info.num_channels,
                    block_len=array_info.chunk_shape.x,
                    file_len=array_info.chunks_per_shard.x,
                    block_type=(
                        wkw.Header.BLOCK_TYPE_LZ4HC
                        if array_info.compression_mode
                        else wkw.Header.BLOCK_TYPE_RAW
                    ),
                ),
            ).close()
        except wkw.wkw.WKWException as e:
            raise ArrayException(f"Exception while creating array {path}") from e
        return WKWArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        return self._wkw_dataset.read(Vec3Int(offset), Vec3Int(shape))

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        self._wkw_dataset.write(Vec3Int(offset), data)

    def ensure_size(
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
            z, y, x = [_extract_num(el) for el in file_path.parts]
            return Vec3Int(x, y, z)

        shard_shape = self.info.shard_shape
        for file_path in self._list_files():
            cube_index = _extract_file_index(file_path)
            cube_offset = cube_index * shard_shape

            yield BoundingBox(cube_offset, shard_shape)

    def close(self) -> None:
        if self._cached_wkw_dataset is not None:
            self._cached_wkw_dataset.close()
            self._cached_wkw_dataset = None

    @property
    def _wkw_dataset(self) -> wkw.Dataset:
        if self._cached_wkw_dataset is None:
            try:
                self._cached_wkw_dataset = wkw.Dataset.open(
                    str(self._path)
                )  # No need to pass the header to the wkw.Dataset
            except wkw.wkw.WKWException as e:
                raise ArrayException(
                    f"Exception while opening WKW array for {self._path}"
                ) from e
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


class ZarrArray(BaseArray):
    data_format = DataFormat.Zarr

    _cached_zarray: Optional[zarr.Array]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_zarray = None

    @classmethod
    def open(cls, path: Path) -> "ZarrArray":
        zarray_path = path / ".zarray"

        if zarray_path.exists() and zarray_path.is_file():
            return cls(path)
        raise ArrayException(
            f"Could not open Zarr array at {path}. `.zarray` not found."
        )

    @property
    def info(self) -> ArrayInfo:
        zarray = self._zarray
        return ArrayInfo(
            data_format=self.data_format,
            num_channels=zarray.shape[0],
            voxel_type=zarray.dtype,
            compression_mode=zarray.compressor is not None,
            chunk_shape=Vec3Int(*zarray.chunks[1:4]) or Vec3Int.full(1),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "ZarrArray":
        assert array_info.data_format == cls.data_format
        assert array_info.chunks_per_shard == Vec3Int.full(
            1
        ), "Zarr storage doesn't support sharding yet"
        zarr.create(
            shape=(array_info.num_channels, 1, 1, 1),
            chunks=(array_info.num_channels,) + array_info.chunk_shape.to_tuple(),
            dtype=array_info.voxel_type,
            compressor=(
                numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
                if array_info.compression_mode
                else None
            ),
            store=_fsstore_from_path(path),
            order="F",
        )
        return ZarrArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        offset = Vec3Int(offset)
        shape = Vec3Int(shape)
        zarray = self._zarray
        with _blosc_disable_threading():
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

    def ensure_size(
        self, new_shape: Vec3IntLike, align_with_shards: bool = True, warn: bool = False
    ) -> None:
        new_shape = Vec3Int(new_shape)
        zarray = self._zarray

        new_shape_tuple = (
            zarray.shape[0],
            max(zarray.shape[1], new_shape.x),
            max(zarray.shape[2], new_shape.y),
            max(zarray.shape[3], new_shape.z),
        )
        if new_shape_tuple != zarray.shape:
            if align_with_shards:
                shard_shape = self.info.shard_shape
                new_shape = new_shape.ceildiv(shard_shape) * shard_shape
                new_shape_tuple = (zarray.shape[0],) + new_shape.to_tuple()

            # Check on-disk for changes to shape
            current_zarray = zarr.open_array(
                store=_fsstore_from_path(self._path), mode="r"
            )
            if zarray.shape != current_zarray.shape:
                warnings.warn(
                    f"[WARNING] While resizing the Zarr array at {self._path}, a differing shape ({zarray.shape} != {current_zarray.shape}) was found in the currently persisted metadata."
                    + "This is likely happening because multiple processes changed the metadata of this array."
                )

            if warn:
                warnings.warn(
                    f"[WARNING] Resizing zarr array from `{zarray.shape}` to `{new_shape_tuple}`."
                )
            zarray.resize(new_shape_tuple)

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        offset = Vec3Int(offset)

        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4

        with _blosc_disable_threading():
            self.ensure_size(offset + Vec3Int(data.shape[1:4]), warn=True)
            zarray = self._zarray
            zarray[
                :,
                offset.x : (offset.x + data.shape[1]),
                offset.y : (offset.y + data.shape[2]),
                offset.z : (offset.z + data.shape[3]),
            ] = data

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        zarray = self._zarray
        chunk_shape = Vec3Int(*zarray.chunks[1:4])
        for key in zarray.store.keys():
            if not key.startswith("."):
                key_parts = [int(p) for p in key.split(zarray._dimension_separator)]
                chunk_idx = Vec3Int(key_parts[1:4])
                yield BoundingBox(topleft=chunk_idx * chunk_shape, size=chunk_shape)

    def close(self) -> None:
        if self._cached_zarray is not None:
            self._cached_zarray = None

    @property
    def _zarray(self) -> zarr.Array:
        if self._cached_zarray is None:
            try:
                self._cached_zarray = zarr.open_array(
                    store=_fsstore_from_path(self._path), mode="a"
                )
            except Exception as e:
                raise ArrayException(
                    f"Exception while opening Zarr array for {self._path}"
                ) from e
        return self._cached_zarray

    @_zarray.deleter
    def _zarray(self) -> None:
        self.close()

    def __del__(self) -> None:
        del self._cached_zarray

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d[""]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d[""] = None
        self.__dict__ = d


class TensorStoreArray(BaseArray):
    data_format = DataFormat.Zarr3

    _cached_array: Optional[ts.TensorStore]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_array = None

    @classmethod
    def open(cls, path: Path) -> "TensorStoreArray":
        try:
            ts.open(
                {
                    "driver": "zarr3",
                    "kvstore": str(path),
                }
            ).result()  # check that everything exists
            return cls(path)
        except Exception as exc:
            raise ArrayException(f"Could not open Zarr3 array at {path}.") from exc

    @property
    def info(self) -> ArrayInfo:
        array = self._array
        array_schema = array.schema
        if (
            len(array_schema.codec) == 1
            and array_schema.codec[0].name == "sharding_indexed"
        ):
            sharding_codec = array_schema.codec[0]
            shard_shape = array_schema.chunk_layout.write_chunk.shape
            chunk_shape = array_schema.chunk_layout.read_chunk.shape
            return ArrayInfo(
                data_format=DataFormat.Zarr3,
                num_channels=array.domain[0].exclusive_max,
                voxel_type=array_schema.dtype,
                compression_mode=True,
                chunk_shape=Vec3Int(chunk_shape[1:4]),
                chunks_per_shard=Vec3Int(shard_shape[1:4]) // Vec3Int(chunk_shape[1:4]),
            )
        return ArrayInfo(
            data_format=DataFormat.Zarr3,
            num_channels=array.domain[0].exclusive_max,
            voxel_type=array_schema.dtype,
            compression_mode=True,
            chunk_shape=Vec3Int(array_schema.chunk_layout.read_chunk.shape[1:4])
            or Vec3Int.full(1),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "TensorStoreArray":
        assert array_info.data_format == DataFormat.Zarr3
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": str(path),
                "metadata": {
                    "data_type": str(array_info.voxel_type),
                    "shape": (array_info.num_channels, 1, 1, 1),
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {
                            "chunk_shape": (array_info.num_channels,)
                            + array_info.shard_shape.to_tuple()
                        },
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "fill_value": 0,
                    "dimension_names": ["c", "x", "y", "z"],
                    "codecs": [
                        {
                            "name": "sharding_indexed",
                            "configuration": {
                                "chunk_shape": (array_info.num_channels,)
                                + array_info.chunk_shape.to_tuple(),
                                "codecs": [
                                    {
                                        "name": "transpose",
                                        "configuration": {"order": "F"},
                                    },
                                    {
                                        "name": "bytes",
                                        "configuration": {"endian": "little"},
                                    },
                                    {
                                        "name": "blosc",
                                        "configuration": {
                                            "cname": "zstd",
                                            "clevel": 5,
                                            "shuffle": "shuffle",
                                            "typesize": array_info.voxel_type.itemsize,
                                        },
                                    },
                                ],
                                # if array_info.compression_mode
                                # else [
                                #     {
                                #         "name": "transpose",
                                #         "configuration": {"order": "F"},
                                #     },
                                #     {
                                #         "name": "bytes",
                                #         "configuration": {"endian": "little"},
                                #     },
                                # ],
                                "index_codecs": [
                                    {
                                        "name": "bytes",
                                        "configuration": {"endian": "little"},
                                    },
                                    {"name": "crc32c"},
                                ],
                            },
                        }
                    ],
                },
                "create": True,
            }
        ).result()
        return TensorStoreArray(path)

    def read(self, offset: Vec3IntLike, shape: Vec3IntLike) -> np.ndarray:
        offset = Vec3Int(offset)
        shape = Vec3Int(shape)
        array = self._array
        data = (
            array[
                :,
                offset.x : (offset.x + shape.x),
                offset.y : (offset.y + shape.y),
                offset.z : (offset.z + shape.z),
            ]
            .read()
            .result()
        )
        return data

    @property
    def current_bbox(self) -> BoundingBox:
        return BoundingBox(
            topleft=self._array.domain.inclusive_min, size=self._array.domain.shape
        )

    def ensure_size(self, bbox: BoundingBox, warn: bool = False) -> None:
        array = self._array

        bbox_domain = ts.IndexDomain(
            [
                ts.Dim(0, self.info.num_channels),
                ts.Dim(bbox.topleft.x, bbox.bottomright.x),
                ts.Dim(bbox.topleft.y, bbox.bottomright.y),
                ts.Dim(bbox.topleft.z, bbox.bottomright.z),
            ]
        )
        new_domain = array.domain.hull(bbox_domain)

        if array.domain != new_domain:
            # Check on-disk for changes to shape
            current_array = ts.open({"driver": "zarr3", "kvstore": self._path}).result()
            if array.domain != current_array.domain:
                warnings.warn(
                    f"[WARNING] While resizing the Zarr array at {self._path}, a differing shape ({array.domain} != {current_array.domain}) was found in the currently persisted metadata."
                    + "This is likely happening because multiple processes changed the metadata of this array."
                )

            if warn:
                warnings.warn(
                    f"[WARNING] Resizing zarr array from `{array.domain}` to `{new_domain}`."
                )
            self._cached_array = array.resize(
                new_domain.inclusive_min,
                new_domain.exclusive_max,
                resize_metadata_only=True,
            )

    def write(self, offset: Vec3IntLike, data: np.ndarray) -> None:
        offset = Vec3Int(offset)

        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4

        self.ensure_size(
            BoundingBox(topleft=offset, size=Vec3Int(data.shape[1:4])), warn=True
        )
        array = self._array
        array[
            :,
            offset.x : (offset.x + data.shape[1]),
            offset.y : (offset.y + data.shape[2]),
            offset.z : (offset.z + data.shape[3]),
        ].write(data).result()

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        raise NotImplementedError

    def close(self) -> None:
        if self._cached_array is not None:
            self._cached_array = None

    @property
    def _array(self) -> ts.TensorStore:
        if self._cached_array is None:
            try:
                _array = ts.open(
                    {"driver": "zarr3", "kvstore": str(self._path)}
                ).result()
                self._cached_array = _array
            except Exception as e:
                raise ArrayException(
                    f"Exception while opening TensorStore for {self._path}"
                ) from e
        return self._cached_array

    @_array.deleter
    def _array(self) -> None:
        self.close()

    def __del__(self) -> None:
        del self._cached_array

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d[""]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d[""] = None
        self.__dict__ = d

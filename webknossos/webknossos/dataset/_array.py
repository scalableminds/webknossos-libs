from typing import (
    TYPE_CHECKING,
)

from .data_format import DataFormat
from ..geometry import NDBoundingBox, VecInt

if TYPE_CHECKING:
    import zarrita
    import zarrita.codecs

import re
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from os.path import relpath
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Optional, Self, Tuple, Type, Union

import numcodecs
import numpy as np
import tensorstore
import wkw
import zarr
from upath import UPath
from zarr.storage import FSStore

from ..geometry import BoundingBox, Vec3Int, Vec3IntLike
from ..utils import is_fs_path, warn_deprecated


def _is_power_of_two(num: int) -> bool:
    return num & (num - 1) == 0


def _fsstore_from_path(path: Path, mode: str = "a") -> FSStore:
    storage_options = {}
    if isinstance(path, UPath) and not is_fs_path(path):
        storage_options = path._kwargs.copy()
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
    shape: VecInt = VecInt(c=1, x=1, y=1, z=1)
    dimension_names: Tuple[str, ...] = ("c", "x", "y", "z")
    axis_order: VecInt = VecInt(c=3, x=2, y=1, z=0)
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
        classes = (WKWArray, Zarr3Array, Zarr2Array)
        for cls in classes:
            try:
                array = cls.open(path)
                return array
            except ArrayException:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
                pass
        raise ArrayException(f"Could not open the array at {path}.")

    @classmethod
    @abstractmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "BaseArray":
        pass

    @abstractmethod
    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def ensure_size(
            self,
            new_bbox: NDBoundingBox,
            align_with_shards: bool = True,
            warn: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def list_bounding_boxes(self) -> Iterator[NDBoundingBox]:
        "The bounding boxes are measured in voxels of the current mag."

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def get_class(data_format: DataFormat) -> Type["BaseArray"]:
        if data_format == DataFormat.WKW:
            return WKWArray
        if data_format == DataFormat.Zarr3:
            return Zarr3Array
        if data_format == DataFormat.Zarr:
            return Zarr2Array
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

        assert array_info.chunk_shape.is_uniform(), f"`chunk_shape` needs to be uniform for WKW storage. Got {array_info.chunk_shape}."
        assert _is_power_of_two(
            array_info.chunk_shape.x
        ), f"`chunk_shape` needs to be a power of 2 for WKW storage. Got {array_info.chunk_shape.x}."
        assert (
                1 <= array_info.chunk_shape.x and array_info.chunk_shape.x <= 32768
        ), f"`chunk_shape` needs to be a value between 1 and 32768 for WKW storage. Got {array_info.chunk_shape.x}."

        assert array_info.chunks_per_shard.is_uniform(), f"`chunks_per_shard` needs to be uniform for WKW storage. Got {array_info.chunks_per_shard}."
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

    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        return self._wkw_dataset.read(Vec3Int(bbox.topleft), Vec3Int(bbox.size))

    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        self._wkw_dataset.write(Vec3Int(bbox.topleft), data)

    def ensure_size(
            self,
            new_bbox: NDBoundingBox,
            align_with_shards: bool = True,
            warn: bool = False,
    ) -> None:
        pass

    def _list_files(self) -> Iterator[Path]:
        return (
            Path(relpath(filename, self._path))
            for filename in self._wkw_dataset.list_files()
        )

    def list_bounding_boxes(self) -> Iterator[NDBoundingBox]:
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


@lru_cache
def _aws_credential_file() -> Path:
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    credentials_file_path = Path(temp_dir_path / "aws_credentials")
    credentials_file_path.touch()


class AWSCredentialManager:
    entries: Dict[int, Tuple[str, str]]
    credentials_file_path: Path

    @classmethod
    @lru_cache
    def singleton(cls) -> "Self":
        return cls()

    def __init__(self) -> None:
        self.entries = {}
        self.credentials_file_path = _aws_credential_file()

    def _dump_credentials(self) -> None:
        self.credentials_file_path.write_text(
            "\n".join(
                [
                    f"[profile-{key_hash}]\naws_access_key_id = {access_key_id}\naws_secret_access_key = {secret_access_key}\n"
                    for key_hash, (
                    access_key_id,
                    secret_access_key,
                ) in self.entries.items()
                ]
            )
        )

    def add(self, access_key_id: str, secret_access_key: str) -> Dict[str, str]:
        key_tuple = (access_key_id, secret_access_key)
        key_hash = hash(key_tuple)
        self.entries[key_hash] = key_tuple
        self._dump_credentials()
        return {
            "profile": f"profile-{key_hash}",
            "filename": str(self.credentials_file_path),
        }


class TensorStoreArray(BaseArray):
    _cached_array: Optional[tensorstore.TensorStore]

    def __init__(
            self, path: Path, _cached_array: Optional[tensorstore.TensorStore] = None
    ):
        super().__init__(path)
        self._cached_array = _cached_array

    @staticmethod
    def _make_kvstore(path: Path) -> Union[str, Dict[str, Union[str, List[str]]]]:
        if is_fs_path(path):
            return {"driver": "file", "path": str(path)}
        elif isinstance(path, UPath) and path.protocol in ("http", "https"):
            return {
                "driver": "http",
                "base_url": str(path),
                "headers": [
                    f"{key}: {value}"
                    for key, value in path.storage_options.get("headers", {}).items()
                ],
            }
        elif isinstance(path, UPath) and path.protocol in ("s3"):
            aws_credential_manager = AWSCredentialManager.singleton()

            return {
                "driver": "s3",
                "base_url": str(path),
                "endpoint_url": path.storage_options.get("client_kwargs", {}).get(
                    "endpoint_url", None
                ),
                "aws_credentials": (
                    aws_credential_manager.add(
                        path.storage_options["key"], path.storage_options["secret"]
                    )
                    if "key" in path.storage_options
                       and "secret" in path.storage_options
                    else None
                ),
            }
        else:
            return str(path)

    @classmethod
    def open(cls, path: Path) -> "TensorStoreArray":
        try:
            uri = cls._make_kvstore(path)
            print(f"{uri=}, {str(cls.data_format)=}, {path=}")
            _array = tensorstore.open(
                {
                    "driver": str(cls.data_format),
                    "kvstore": uri,
                },
                open=True,
                create=True
            ).result()  # check that everything exists
            return cls(path, _array)
        except Exception as exc:
            raise ArrayException(f"Could not open array at {uri}.") from exc

    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        offset = Vec3Int(bbox.topleft)
        shape = Vec3Int(bbox.size)
        array = self._array

        requested_domain = tensorstore.IndexDomain(
            [
                tensorstore.Dim(0, self.info.num_channels),
                tensorstore.Dim(offset.x, offset.x + shape.x),
                tensorstore.Dim(offset.y, offset.y + shape.y),
                tensorstore.Dim(offset.z, offset.z + shape.z),
            ]
        )
        available_domain = requested_domain.intersect(array.domain)
        if available_domain != requested_domain:
            # needs padding
            out = np.zeros(
                requested_domain.shape, dtype=array.dtype.numpy_dtype, order="F"
            )
            data = array[available_domain].read(order="F").result()
            out[
            : data.shape[0],
            : data.shape[1],
            : data.shape[2],
            : data.shape[3],
            ] = data
            return out

        return array[requested_domain].read(order="F").result()

    def ensure_size(self, bbox: BoundingBox, warn: bool = False) -> None:
        array = self._array

        bbox_domain = tensorstore.IndexDomain(
            [
                tensorstore.Dim(0, self.info.num_channels, implicit_upper=True),
                tensorstore.Dim(
                    bbox.topleft.x, bbox.bottomright.x, implicit_upper=True
                ),
                tensorstore.Dim(
                    bbox.topleft.y, bbox.bottomright.y, implicit_upper=True
                ),
                tensorstore.Dim(
                    bbox.topleft.z, bbox.bottomright.z, implicit_upper=True
                ),
            ]
        )
        new_domain = array.domain.hull(bbox_domain)

        if array.domain != new_domain:
            # Check on-disk for changes to shape
            current_array = tensorstore.open(
                {
                    "driver": str(self.data_format),
                    "kvstore": self._make_kvstore(self._path),
                }
            ).result()
            if array.domain != current_array.domain:
                warnings.warn(
                    f"[WARNING] While resizing the Zarr array at {self._path}, a differing shape ({array.domain} != {current_array.domain}) was found in the currently persisted metadata."
                    + "This is likely happening because multiple processes changed the metadata of this array."
                )

            if warn:
                warnings.warn(
                    f"[WARNING] Resizing Zarr array from `{array.domain}` to `{new_domain}`."
                )
            self._cached_array = array.resize(
                inclusive_min=None,
                exclusive_max=new_domain.exclusive_max,
                resize_metadata_only=True,
            ).result()

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
        offset.x: (offset.x + data.shape[1]),
        offset.y: (offset.y + data.shape[2]),
        offset.z: (offset.z + data.shape[3]),
        ].write(data).result()

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        raise NotImplementedError

    def close(self) -> None:
        if self._cached_array is not None:
            self._cached_array = None

    @property
    def _array(self) -> tensorstore.TensorStore:
        if self._cached_array is None:
            try:
                self._cached_array = tensorstore.open(
                    {
                        "driver": str(self.data_format),
                        "kvstore": self._make_kvstore(self._path),
                    }
                ).result()
            except Exception as e:
                raise ArrayException(
                    f"Exception while opening array for {self._make_kvstore(self._path)}"
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


class Zarr3Array(TensorStoreArray):
    data_format = DataFormat.Zarr3

    @property
    def info(self) -> ArrayInfo:
        array = self._array
        array_codecs = array.codec.to_json()["codecs"]
        if len(array_codecs) == 1 and array_codecs[0]["name"] == "sharding_indexed":
            shard_shape = array.chunk_layout.write_chunk.shape
            chunk_shape = array.chunk_layout.read_chunk.shape
            return ArrayInfo(
                data_format=DataFormat.Zarr3,
                num_channels=array.domain[0].exclusive_max,
                voxel_type=array.dtype.numpy_dtype,
                compression_mode=self._has_compression_codecs(
                    array_codecs[0]["configuration"]["codecs"]
                ),
                chunk_shape=Vec3Int(chunk_shape[1:4]),
                chunks_per_shard=Vec3Int(shard_shape[1:4]) // Vec3Int(chunk_shape[1:4]),
            )
        return ArrayInfo(
            data_format=DataFormat.Zarr3,
            num_channels=array.domain[0].exclusive_max,
            voxel_type=array.dtype.numpy_dtype,
            compression_mode=self._has_compression_codecs(array_codecs),
            chunk_shape=Vec3Int(array.chunk_layout.read_chunk.shape[1:4]),
            chunks_per_shard=Vec3Int.full(1),
        )

    @staticmethod
    def _has_compression_codecs(codecs: List[Dict[str, str]]) -> bool:
        return any(c["name"] in ("blosc", "gzip", "zstd") for c in codecs)

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "Zarr3Array":
        assert array_info.data_format == cls.data_format
        _array = tensorstore.open(
            {
                "driver": str(cls.data_format),
                "kvstore": cls._make_kvstore(path),
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
                                "codecs": (
                                    [
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
                                    ]
                                    if array_info.compression_mode
                                    else [
                                        {
                                            "name": "transpose",
                                            "configuration": {"order": "F"},
                                        },
                                        {
                                            "name": "bytes",
                                            "configuration": {"endian": "little"},
                                        },
                                    ]
                                ),
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
        return cls(path, _array)

    def ensure_size(self, bbox: BoundingBox, warn: bool = False) -> None:
        super().ensure_size(
            bbox.align_with_mag(self.info.shard_shape, ceil=True), warn=warn
        )


class Zarr2Array(TensorStoreArray):
    data_format = DataFormat.Zarr

    @property
    def info(self) -> ArrayInfo:
        array = self._array
        return ArrayInfo(
            data_format=DataFormat.Zarr,
            num_channels=array.domain[0].exclusive_max,
            voxel_type=array.dtype.numpy_dtype,
            compression_mode=array.codec is not None,
            chunk_shape=Vec3Int(array.chunk_layout.read_chunk.shape[1:4]),
            chunks_per_shard=Vec3Int.full(1),
        )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "Zarr2Array":
        assert array_info.data_format == DataFormat.Zarr
        assert array_info.chunks_per_shard == Vec3Int.full(
            1
        ), "Zarr2 storage doesn't support sharding yet"
        _array = tensorstore.open(
            {
                "driver": "zarr",
                "kvstore": cls._make_kvstore(path),
                "metadata": {
                    "shape": (array_info.num_channels, 1, 1, 1),
                    "chunks": (array_info.num_channels,)
                              + array_info.shard_shape.to_tuple(),
                    "dtype": array_info.voxel_type.str,
                    "fill_value": 0,
                    "order": "F",
                    "compressor": (
                        {
                            "id": "blosc",
                            "cname": "zstd",
                            "clevel": 5,
                            "shuffle": 1,
                        }
                        if array_info.compression_mode
                        else None
                    ),
                    "filters": None,
                    "dimension_separator": ".",
                },
                "create": True,
            }
        ).result()
        return cls(path, _array)


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

    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        shape = bbox.size
        zarray = self._zarray
        with _blosc_disable_threading():
            data = zarray[(slice(None),) + bbox.to_slices()]

        shape_with_channels = (self.info.num_channels,) + shape.to_tuple()
        if data.shape not in (shape, shape_with_channels):
            padded_data = np.zeros(shape_with_channels, dtype=data.dtype)
            padded_data[
            :,
            0: data.shape[1],
            0: data.shape[2],
            0: data.shape[3],
            ] = data
            data = padded_data
        return data

    def ensure_size(
            self,
            new_bbox: NDBoundingBox,
            align_with_shards: bool = True,
            warn: bool = False,
    ) -> None:
        new_shape = VecInt(new_bbox.size, axes=new_bbox.axes)
        zarray = self._zarray

        new_shape_tuple = (zarray.shape[0],) + tuple(
            (
                max(zarray.shape[i + 1], new_shape[i])
                if len(zarray.shape) > i
                else new_shape[i]
            )
            for i in range(len(new_shape))
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

    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        """Writes a ZarrArray. If offset and bbox are given, the bbox is preferred to enable writing of n-dimensional data."""

        # If data is 3-dimensional, it is assumed that num_channels=1.
        if data.ndim == 3:
            data = data.reshape((1,) + data.shape)
        assert data.ndim == 4

        with _blosc_disable_threading():
            self.ensure_size(bbox, warn=True)
            zarray = self._zarray
            index_tuple = (slice(None),) + bbox.to_slices()

            zarray[index_tuple] = data

    def list_bounding_boxes(self) -> Iterator[NDBoundingBox]:
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
        del d["_cached_zarray"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_zarray"] = None
        self.__dict__ = d


class ZarritaArray(BaseArray):
    data_format = DataFormat.Zarr3

    _cached_zarray: Optional[Union["zarrita.Array", "zarrita.ArrayV2"]]

    def __init__(self, path: Path):
        super().__init__(path)
        self._cached_zarray = None

    @classmethod
    def open(cls, path: Path) -> "ZarritaArray":
        from zarrita import Array

        try:
            Array.open_auto(store=path)  # check that everything exists
            return cls(path)
        except Exception as exc:
            raise ArrayException(f"Could not open Zarr array at {path}.") from exc

    @staticmethod
    def _has_compression_codecs(codecs: List["zarrita.codecs.Codec"]) -> bool:
        from zarrita.codecs import BloscCodec, GzipCodec, ZstdCodec

        return any(
            isinstance(c, BloscCodec)
            or isinstance(c, GzipCodec)
            or isinstance(c, ZstdCodec)
            for c in codecs
        )

    @property
    def info(self) -> ArrayInfo:
        from zarrita import Array
        from zarrita.sharding import ShardingCodec

        zarray = self._zarray
        if (names := getattr(zarray.metadata, "dimension_names", None)) is None:
            dimension_names = ("c", "x", "y", "z")
        else:
            dimension_names = names
        x_index, y_index, z_index = (
            dimension_names.index("x"),
            dimension_names.index("y"),
            dimension_names.index("z"),
        )
        if isinstance(zarray, Array):
            if len(zarray.codec_pipeline.codecs) == 1 and isinstance(
                    zarray.codec_pipeline.codecs[0], ShardingCodec
            ):
                sharding_codec = zarray.codec_pipeline.codecs[0]
                shard_shape = zarray.metadata.chunk_grid.configuration.chunk_shape
                chunk_shape = sharding_codec.configuration.chunk_shape
                return ArrayInfo(
                    data_format=DataFormat.Zarr3,
                    num_channels=zarray.metadata.shape[0],
                    voxel_type=zarray.metadata.dtype,
                    compression_mode=self._has_compression_codecs(
                        sharding_codec.codec_pipeline.codecs
                    ),
                    chunk_shape=Vec3Int(
                        chunk_shape[x_index], chunk_shape[y_index], chunk_shape[z_index]
                    ),
                    chunks_per_shard=Vec3Int(
                        Vec3Int(
                            shard_shape[x_index],
                            shard_shape[y_index],
                            shard_shape[z_index],
                        )
                        // Vec3Int(
                            chunk_shape[x_index],
                            chunk_shape[y_index],
                            chunk_shape[z_index],
                        )
                    ),
                    dimension_names=dimension_names,
                )
            chunk_shape = zarray.metadata.chunk_grid.configuration.chunk_shape
            return ArrayInfo(
                data_format=DataFormat.Zarr3,
                num_channels=zarray.metadata.shape[0],
                voxel_type=zarray.metadata.dtype,
                compression_mode=self._has_compression_codecs(
                    zarray.codec_pipeline.codecs
                ),
                chunk_shape=Vec3Int(
                    chunk_shape[x_index], chunk_shape[y_index], chunk_shape[z_index]
                )
                            or Vec3Int.full(1),
                chunks_per_shard=Vec3Int.full(1),
                dimension_names=dimension_names,
            )
        else:
            return ArrayInfo(
                data_format=DataFormat.Zarr,
                num_channels=zarray.metadata.shape[0],
                voxel_type=zarray.metadata.dtype,
                compression_mode=zarray.metadata.compressor is not None,
                chunk_shape=Vec3Int(
                    zarray.metadata.chunks[x_index],
                    zarray.metadata.chunks[y_index],
                    zarray.metadata.chunks[z_index],
                )
                            or Vec3Int.full(1),
                chunks_per_shard=Vec3Int.full(1),
                dimension_names=dimension_names,
            )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "ZarritaArray":
        import zarrita.codecs
        from zarrita import Array, ArrayV2

        assert array_info.data_format in (DataFormat.Zarr, DataFormat.Zarr3)
        if array_info.data_format == DataFormat.Zarr3:
            chunk_shape = (array_info.num_channels,) + tuple(
                getattr(array_info.chunk_shape, axis, 1)
                for axis in array_info.dimension_names[1:]
            )
            shard_shape = (array_info.num_channels,) + tuple(
                getattr(array_info.shard_shape, axis, 1)
                for axis in array_info.dimension_names[1:]
            )
            Array.create(
                store=path,
                shape=array_info.shape,
                chunk_shape=shard_shape,
                chunk_key_encoding=("default", "/"),
                dtype=array_info.voxel_type,
                dimension_names=array_info.dimension_names,
                codecs=[
                    zarrita.codecs.sharding_codec(
                        chunk_shape=chunk_shape,
                        codecs=(
                            [
                                zarrita.codecs.transpose_codec(array_info.axis_order),
                                zarrita.codecs.bytes_codec(),
                                zarrita.codecs.blosc_codec(
                                    typesize=array_info.voxel_type.itemsize
                                ),
                            ]
                            if array_info.compression_mode
                            else [
                                zarrita.codecs.transpose_codec(array_info.axis_order),
                                zarrita.codecs.bytes_codec(),
                            ]
                        ),
                    )
                ],
            )
        else:
            ArrayV2.create(
                store=path,
                shape=(array_info.shape),
                chunks=(array_info.num_channels,)
                       + tuple(
                    getattr(array_info.chunk_shape, axis, 1)
                    for axis in array_info.dimension_names[1:]
                ),
                # The chunk shape consists of the number of channels, the x, y, and z dimensions of a chunk, and 1 for all other dimensions.
                dtype=array_info.voxel_type,
                compressor=(
                    {"id": "blosc", "cname": "zstd", "clevel": 5}
                    if array_info.compression_mode
                    else None
                ),
                order="F",
                dimension_separator=".",
            )
        return ZarritaArray(path)

    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        shape = bbox.size.to_tuple()
        zarray = self._zarray
        slice_tuple = (slice(None),) + bbox.to_slices()
        with _blosc_disable_threading():
            data = zarray[slice_tuple]

        shape_with_channels = (self.info.num_channels,) + shape
        if data.shape != shape_with_channels:
            data_slice_tuple = tuple(slice(0, size) for size in data.shape)
            padded_data = np.zeros(shape_with_channels, dtype=zarray.metadata.dtype)
            padded_data[data_slice_tuple] = data
            data = padded_data
        return data

    def ensure_size(
            self,
            new_bbox: NDBoundingBox,
            align_with_shards: bool = True,
            warn: bool = False,
    ) -> None:
        zarray = self._zarray

        new_bbox = new_bbox.with_bottomright(
            (
                max(zarray.metadata.shape[i + 1], new_bbox.bottomright[i])
                for i in range(len(new_bbox))
            )
        )
        new_shape_tuple = (zarray.metadata.shape[0],) + tuple(new_bbox.bottomright)
        if new_shape_tuple != zarray.metadata.shape:
            if align_with_shards:
                shard_shape = self.info.shard_shape
                new_aligned_bbox = new_bbox.with_bottomright_xyz(
                    new_bbox.bottomright_xyz.ceildiv(shard_shape) * shard_shape
                )
                new_shape_tuple = (
                                      zarray.metadata.shape[0],
                                  ) + new_aligned_bbox.bottomright.to_tuple()

            # Check on-disk for changes to shape
            current_zarray = zarray.open(self._path)
            if zarray.metadata.shape != current_zarray.metadata.shape:
                warnings.warn(
                    f"[WARNING] While resizing the Zarr array at {self._path}, a differing shape ({zarray.metadata.shape} != {current_zarray.metadata.shape}) was found in the currently persisted metadata."
                    + "This is likely happening because multiple processes changed the metadata of this array."
                )

            if warn:
                warnings.warn(
                    f"[WARNING] Resizing zarr array from `{zarray.metadata.shape}` to `{new_shape_tuple}`."
                )
            self._cached_zarray = zarray.resize(new_shape_tuple)

    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        if data.ndim == len(bbox):
            # the bbox does not include the channels, if data and bbox have the same size there is only 1 channel
            data = data.reshape((1,) + data.shape)

        assert data.ndim == len(bbox) + 1

        with _blosc_disable_threading():
            self.ensure_size(bbox, warn=True)
            zarray = self._zarray
            index_tuple = (slice(None),) + bbox.to_slices()

            zarray[index_tuple] = data

    def list_bounding_boxes(self) -> Iterator[NDBoundingBox]:
        raise NotImplementedError

    def close(self) -> None:
        if self._cached_zarray is not None:
            self._cached_zarray = None

    @property
    def _zarray(self) -> Union["zarrita.Array", "zarrita.ArrayV2"]:
        from zarrita import Array, runtime_configuration

        if self._cached_zarray is None:
            try:
                zarray = Array.open_auto(
                    store=self._path,
                    runtime_configuration=runtime_configuration("F"),
                )
                self._cached_zarray = zarray
            except Exception as e:
                raise ArrayException(
                    f"Exception while opening Zarrita array for {self._path}"
                ) from e
        return self._cached_zarray

    @_zarray.deleter
    def _zarray(self) -> None:
        self.close()

    def __del__(self) -> None:
        del self._cached_zarray

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_zarray"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_zarray"] = None
        self.__dict__ = d

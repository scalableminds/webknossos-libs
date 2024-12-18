import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from os.path import relpath
from pathlib import Path
from tempfile import mkdtemp
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urlparse

import numpy as np
import tensorstore
import wkw
from typing_extensions import Self
from upath import UPath

from ..geometry import BoundingBox, NDBoundingBox, Vec3Int, VecInt
from ..utils import is_fs_path, warn_deprecated
from .data_format import DataFormat


def _is_power_of_two(num: int) -> bool:
    return num & (num - 1) == 0


class ArrayException(Exception):
    pass


@dataclass
class ArrayInfo:
    data_format: DataFormat
    num_channels: int
    voxel_type: np.dtype
    chunk_shape: Vec3Int
    shard_shape: Vec3Int
    shape: VecInt = VecInt(c=1, x=1, y=1, z=1)
    dimension_names: Tuple[str, ...] = ("c", "x", "y", "z")
    axis_order: VecInt = VecInt(c=3, x=2, y=1, z=0)
    compression_mode: bool = False

    @property
    def shard_size(self) -> Vec3Int:
        warn_deprecated("shard_size", "shard_shape")
        return self.shard_shape

    @property
    def chunks_per_shard(self) -> Vec3Int:
        return self.shard_shape // self.chunk_shape

    @property
    def ndim(self) -> int:
        return len(self.shape)


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
                array = cls.open(path)  # type: ignore[attr-defined]
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
            shard_shape=Vec3Int.full(
                header.file_len,
            )
            * Vec3Int.full(header.block_len),
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


class AWSCredentialManager:
    entries: Dict[int, Tuple[str, str]]
    credentials_file_path: Path

    def __init__(self, credentials_file_path: Path) -> None:
        self.entries = {}
        self.credentials_file_path = credentials_file_path

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


@lru_cache
def _aws_credential_file() -> Path:
    temp_dir_path = Path(mkdtemp())
    credentials_file_path = Path(temp_dir_path / "aws_credentials")
    credentials_file_path.touch()
    return credentials_file_path


_aws_credential_manager = AWSCredentialManager(_aws_credential_file())


class TensorStoreArray(BaseArray):
    _cached_array: Optional[tensorstore.TensorStore]

    def __init__(
        self, path: Path, _cached_array: Optional[tensorstore.TensorStore] = None
    ):
        super().__init__(path)
        self._cached_array = _cached_array

    @staticmethod
    def _get_array_dimensions(
        array: tensorstore.TensorStore,
    ) -> tuple[tuple[str, ...], Vec3Int, Vec3Int, int, VecInt]:
        axes = array.domain.labels
        array_chunk_shape = array.chunk_layout.read_chunk.shape
        array_shard_shape = array.chunk_layout.write_chunk.shape

        chunk_shape = Vec3Int.ones()
        shard_shape = Vec3Int.ones()
        num_channels = 1
        dimension_names: tuple[str, ...] = ()
        if all(a == "" for a in axes):
            if len(array.shape) == 2:
                dimension_names = ("x", "y")
                chunk_shape = Vec3Int(array_chunk_shape[0], array_chunk_shape[1], 1)
                shard_shape = Vec3Int(array_shard_shape[0], array_shard_shape[1], 1)
            elif len(array.shape) == 3:
                dimension_names = ("x", "y", "z")
                chunk_shape = Vec3Int(
                    array_chunk_shape[0], array_chunk_shape[1], array_chunk_shape[2]
                )
                shard_shape = Vec3Int(
                    array_shard_shape[0], array_shard_shape[1], array_shard_shape[2]
                )
            elif len(array.shape) == 4:
                dimension_names = ("c", "x", "y", "z")
                num_channels = array.domain[0].exclusive_max
                chunk_shape = Vec3Int(
                    array_chunk_shape[1], array_chunk_shape[2], array_chunk_shape[3]
                )
                shard_shape = Vec3Int(
                    array_shard_shape[1], array_shard_shape[2], array_shard_shape[3]
                )
            else:
                raise ArrayException(
                    f"Zarr3 arrays with less than 2 and more than 4 dimensions are not supported without explicit `dimension_names`. Got {len(array.shape)} dimensions."
                )
        else:
            dimension_names = axes
            if "x" in dimension_names and "y" in dimension_names:
                x_index, y_index = (
                    dimension_names.index("x"),
                    dimension_names.index("y"),
                )
                if "z" in dimension_names:
                    z_index = dimension_names.index("z")
                    chunk_shape = Vec3Int(
                        array_chunk_shape[x_index],
                        array_chunk_shape[y_index],
                        array_chunk_shape[z_index],
                    )
                    shard_shape = Vec3Int(
                        array_shard_shape[x_index],
                        array_shard_shape[y_index],
                        array_shard_shape[z_index],
                    )
                else:
                    chunk_shape = Vec3Int(
                        array_chunk_shape[x_index],
                        array_chunk_shape[y_index],
                        1,
                    )
                    shard_shape = Vec3Int(
                        array_shard_shape[x_index],
                        array_shard_shape[y_index],
                        1,
                    )
                if "c" in dimension_names:
                    c_index = dimension_names.index("c")
                    num_channels = array.domain[c_index].exclusive_max
            else:
                raise ArrayException(
                    f"Zarr3 arrays without x and y dimensions are not supported. Got {axes} dimensions."
                )

        shape = array.domain.exclusive_max
        if "c" not in dimension_names:
            shape = (num_channels,) + shape
            dimension_names = ("c",) + dimension_names

        return (
            dimension_names,
            chunk_shape,
            shard_shape,
            num_channels,
            VecInt(shape, axes=dimension_names),
        )

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
            parsed_url = urlparse(str(path))
            kvstore_spec: dict[str, Any] = {
                "driver": "s3",
                "path": parsed_url.path,
                "bucket": parsed_url.netloc,
            }
            if endpoint_url := path.storage_options.get("client_kwargs", {}).get(
                "endpoint_url", None
            ):
                kvstore_spec["endpoint"] = endpoint_url
            if "key" in path.storage_options and "secret" in path.storage_options:
                kvstore_spec["aws_credentials"] = _aws_credential_manager.add(
                    path.storage_options["key"], path.storage_options["secret"]
                )
            return kvstore_spec
        else:
            return {
                "driver": "file",
                "path": str(path),
            }

    @classmethod
    def open(cls, path: Path) -> "TensorStoreArray":
        classes = (Zarr3Array, Zarr2Array)
        for _cls in classes:
            try:
                array = _cls.open(path)
                return array
            except ArrayException:  # noqa: PERF203
                pass
        raise ArrayException(f"Could not open the array at {path}.")

    @classmethod
    def _open(cls, path: Path) -> Self:
        try:
            uri = cls._make_kvstore(path)
            _array = tensorstore.open(
                {
                    "driver": str(cls.data_format),
                    "kvstore": uri,
                },
                open=True,
                create=True,
            ).result()  # check that everything exists
            return cls(path, _array)
        except Exception as exc:
            raise ArrayException(f"Could not open array at {uri}.") from exc

    def read(self, bbox: NDBoundingBox) -> np.ndarray:
        array = self._array

        has_channel_dimension = len(self.info.shape) == len(array.domain)

        if not has_channel_dimension:
            requested_domain = tensorstore.IndexDomain(
                bbox.ndim,
                inclusive_min=bbox.topleft.to_tuple(),
                shape=bbox.size.to_tuple(),
            )
            available_domain = requested_domain.intersect(array.domain)
        else:
            requested_domain = tensorstore.IndexDomain(
                bbox.ndim + 1,
                inclusive_min=(0,) + bbox.topleft.to_tuple(),
                shape=(self.info.num_channels,) + bbox.size.to_tuple(),
            )
            available_domain = requested_domain.intersect(array.domain)

        if available_domain != requested_domain:
            # needs padding
            out = np.zeros(
                requested_domain.shape, dtype=array.dtype.numpy_dtype, order="F"
            )
            data = array[available_domain].read(order="F").result()
            out[tuple(slice(0, data.shape[i]) for i in range(len(data.shape)))] = data
            if not has_channel_dimension:
                out = np.expand_dims(out, 0)
            return out

        out = array[requested_domain].read(order="F").result()
        if not has_channel_dimension:
            out = np.expand_dims(out, 0)
        return out

    def ensure_size(
        self,
        new_bbox: NDBoundingBox,
        align_with_shards: bool = True,
        warn: bool = False,
    ) -> None:
        array = self._array

        new_bbox = new_bbox.with_bottomright(
            (
                max(array.domain.exclusive_max[i + 1], new_bbox.bottomright[i])
                for i in range(len(new_bbox))
            )
        )
        new_domain = tensorstore.IndexDomain(
            new_bbox.ndim + 1,
            shape=(self.info.num_channels,) + new_bbox.bottomright.to_tuple(),
            implicit_upper_bounds=tuple(True for _ in range(new_bbox.ndim + 1)),
            labels=array.domain.labels,
        )
        if new_domain != array.domain:
            if align_with_shards:
                shard_shape = self.info.shard_shape
                new_aligned_bbox = new_bbox.with_bottomright_xyz(
                    new_bbox.bottomright_xyz.ceildiv(shard_shape) * shard_shape
                )
                new_domain = tensorstore.IndexDomain(
                    new_aligned_bbox.ndim + 1,
                    shape=(self.info.num_channels,)
                    + new_aligned_bbox.bottomright.to_tuple(),
                    implicit_upper_bounds=tuple(
                        True for _ in range(new_aligned_bbox.ndim + 1)
                    ),
                    labels=array.domain.labels,
                )

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
                expand_only=True,
            ).result()

    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        if data.ndim == len(bbox):
            # the bbox does not include the channels, if data and bbox have the same size there is only 1 channel
            data = data.reshape((1,) + data.shape)

        assert data.ndim == len(bbox) + 1

        self.ensure_size(bbox, warn=True)
        array = self._array

        requested_domain = tensorstore.IndexDomain(
            bbox.ndim + 1,
            inclusive_min=(0,) + bbox.topleft.to_tuple(),
            shape=(self.info.num_channels,) + bbox.size.to_tuple(),
        )
        array[requested_domain].write(data).result()

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

    @classmethod
    def open(cls, path: Path) -> "Zarr3Array":
        return cls._open(path)

    @property
    def info(self) -> ArrayInfo:
        array = self._array
        array_codecs = array.codec.to_json()["codecs"]

        dimension_names, chunk_shape, shard_shape, num_channels, shape = (
            self._get_array_dimensions(array)
        )

        if len(array_codecs) == 1 and array_codecs[0]["name"] == "sharding_indexed":
            return ArrayInfo(
                data_format=DataFormat.Zarr3,
                num_channels=num_channels,
                voxel_type=array.dtype.numpy_dtype,
                compression_mode=self._has_compression_codecs(
                    array_codecs[0]["configuration"]["codecs"]
                ),
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                shape=shape,
                dimension_names=dimension_names,
            )
        return ArrayInfo(
            data_format=DataFormat.Zarr3,
            num_channels=num_channels,
            voxel_type=array.dtype.numpy_dtype,
            compression_mode=self._has_compression_codecs(array_codecs),
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            shape=shape,
            dimension_names=dimension_names,
        )

    @staticmethod
    def _has_compression_codecs(codecs: List[Dict[str, str]]) -> bool:
        return any(c["name"] in ("blosc", "gzip", "zstd") for c in codecs)

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "Zarr3Array":
        assert array_info.data_format == cls.data_format
        chunk_shape = (array_info.num_channels,) + tuple(
            getattr(array_info.chunk_shape, axis, 1)
            for axis in array_info.dimension_names[1:]
        )
        shard_shape = (array_info.num_channels,) + tuple(
            getattr(array_info.shard_shape, axis, 1)
            for axis in array_info.dimension_names[1:]
        )
        shape = tuple(
            shape_a - (shape_a % shard_shape_a)
            for shape_a, shard_shape_a in zip(array_info.shape, shard_shape)
        )
        _array = tensorstore.open(
            {
                "driver": str(cls.data_format),
                "kvstore": cls._make_kvstore(path),
                "metadata": {
                    "data_type": str(array_info.voxel_type),
                    "shape": shape,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": shard_shape},
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "fill_value": 0,
                    "dimension_names": array_info.dimension_names,
                    "codecs": [
                        {
                            "name": "sharding_indexed",
                            "configuration": {
                                "chunk_shape": chunk_shape,
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


class Zarr2Array(TensorStoreArray):
    data_format = DataFormat.Zarr

    @classmethod
    def open(cls, path: Path) -> "Zarr2Array":
        return cls._open(path)

    @property
    def info(self) -> ArrayInfo:
        array = self._array
        dimension_names, chunk_shape, shard_shape, num_channels, shape = (
            self._get_array_dimensions(array)
        )

        return ArrayInfo(
            data_format=DataFormat.Zarr,
            num_channels=num_channels,
            voxel_type=array.dtype.numpy_dtype,
            compression_mode=array.codec.to_json()["compressor"] is not None,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            shape=shape,
        )

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "Zarr2Array":
        assert array_info.data_format == DataFormat.Zarr
        assert array_info.chunks_per_shard == Vec3Int.full(
            1
        ), "Zarr2 storage doesn't support sharding yet"
        chunk_shape = (array_info.num_channels,) + tuple(
            getattr(array_info.chunk_shape, axis, 1)
            for axis in array_info.dimension_names[1:]
        )
        shape = tuple(
            shape_a - (shape_a % chunk_shape_a)
            for shape_a, chunk_shape_a in zip(array_info.shape, chunk_shape)
        )
        _array = tensorstore.open(
            {
                "driver": "zarr",
                "kvstore": cls._make_kvstore(path),
                "metadata": {
                    "shape": shape,
                    "chunks": chunk_shape,
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

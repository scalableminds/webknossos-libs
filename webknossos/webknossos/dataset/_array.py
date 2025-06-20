import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from logging import getLogger
from os.path import relpath
from pathlib import Path
from tempfile import mkdtemp
from typing import (
    Any,
    Literal,
)
from urllib.parse import urlparse

import numpy as np
import tensorstore
import wkw
from typing_extensions import Self
from upath import UPath

from ..geometry import BoundingBox, NDBoundingBox, Vec3Int, VecInt
from ..utils import call_with_retries, is_fs_path
from .data_format import DataFormat

logger = getLogger(__name__)

TS_CONTEXT = tensorstore.Context()


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
    dimension_names: tuple[str, ...] = ("c", "x", "y", "z")
    axis_order: VecInt = VecInt(c=3, x=2, y=1, z=0)
    compression_mode: bool = False

    @property
    def chunks_per_shard(self) -> Vec3Int:
        return self.shard_shape // self.chunk_shape

    @property
    def ndim(self) -> int:
        return len(self.shape)


class BaseArray(ABC):
    data_format = DataFormat.WKW

    _path: UPath

    def __init__(self, path: Path):
        self._path = UPath(path)

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
    def resize(self, new_bbox: NDBoundingBox) -> None:
        pass

    @abstractmethod
    def list_bounding_boxes(self) -> Iterator[NDBoundingBox]:
        "The bounding boxes are measured in voxels of the current mag."

    @abstractmethod
    def close(self) -> None:
        pass

    @staticmethod
    def get_class(data_format: DataFormat) -> type["BaseArray"]:
        if data_format == DataFormat.WKW:
            return WKWArray
        if data_format == DataFormat.Zarr3:
            return Zarr3Array
        if data_format == DataFormat.Zarr:
            return Zarr2Array
        raise ValueError(f"Array format `{data_format}` is invalid.")


class WKWArray(BaseArray):
    data_format = DataFormat.WKW

    _cached_wkw_dataset: wkw.Dataset | None

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

        assert array_info.chunk_shape.is_uniform(), (
            f"`chunk_shape` needs to be uniform for WKW storage. Got {array_info.chunk_shape}."
        )
        assert _is_power_of_two(array_info.chunk_shape.x), (
            f"`chunk_shape` needs to be a power of 2 for WKW storage. Got {array_info.chunk_shape.x}."
        )
        assert 1 <= array_info.chunk_shape.x and array_info.chunk_shape.x <= 32768, (
            f"`chunk_shape` needs to be a value between 1 and 32768 for WKW storage. Got {array_info.chunk_shape.x}."
        )

        assert array_info.chunks_per_shard.is_uniform(), (
            f"`chunks_per_shard` needs to be uniform for WKW storage. Got {array_info.chunks_per_shard}."
        )
        assert _is_power_of_two(array_info.chunks_per_shard.x), (
            f"`chunks_per_shard` needs to be a power of 2 for WKW storage. Got {array_info.chunks_per_shard.x}."
        )
        assert (
            1 <= array_info.chunks_per_shard.x
            and array_info.chunks_per_shard.x <= 32768
        ), (
            f"`chunks_per_shard` needs to be a value between 1 and 32768 for WKW storage. Got {array_info.chunks_per_shard.x}."
        )

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

    def resize(self, new_bbox: NDBoundingBox) -> None:
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

    def __getstate__(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_wkw_dataset"]
        return d

    def __setstate__(self, d: dict[str, Any]) -> None:
        d["_cached_wkw_dataset"] = None
        self.__dict__ = d


class AWSCredentialManager:
    entries: dict[int, tuple[str, str]]
    folder_path: Path

    def __init__(self, folder_path: Path) -> None:
        self.entries = {}
        self.folder_path = folder_path

        self.credentials_file_path.touch()
        self.config_file_path.write_text("[default]\n")

    @property
    def credentials_file_path(self) -> Path:
        return self.folder_path / "credentials"

    @property
    def config_file_path(self) -> Path:
        return self.folder_path / "config"

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

    def add(self, access_key_id: str, secret_access_key: str) -> dict[str, str]:
        key_tuple = (access_key_id, secret_access_key)
        key_hash = hash(key_tuple)
        self.entries[key_hash] = key_tuple
        self._dump_credentials()
        return {
            "type": "profile",
            "profile": f"profile-{key_hash}",
            "config_file": str(self.config_file_path),
            "credentials_file": str(self.credentials_file_path),
        }


@lru_cache
def _aws_credential_folder() -> Path:
    return Path(mkdtemp())


_aws_credential_manager = AWSCredentialManager(_aws_credential_folder())


class TensorStoreArray(BaseArray):
    _cached_array: tensorstore.TensorStore | None

    def __init__(
        self, path: Path, _cached_array: tensorstore.TensorStore | None = None
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
            }
            if endpoint_url := path.storage_options.get("client_kwargs", {}).get(
                "endpoint_url", None
            ):
                kvstore_spec["endpoint"] = endpoint_url
            if "key" in path.storage_options and "secret" in path.storage_options:
                kvstore_spec["aws_credentials"] = _aws_credential_manager.add(
                    path.storage_options["key"], path.storage_options["secret"]
                )
            else:
                kvstore_spec["aws_credentials"] = {"type": "default"}
            return kvstore_spec
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
    def _open(cls, path: UPath) -> Self:
        try:
            uri = cls._make_kvstore(path)
            _array = call_with_retries(
                lambda: tensorstore.open(
                    {
                        "driver": str(cls.data_format),
                        "kvstore": uri,
                    },
                    open=True,
                    create=False,
                    context=TS_CONTEXT,
                ).result(),
                description="Opening tensorstore array",
            )  # check that everything exists
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

        data = call_with_retries(
            lambda: array[available_domain].read(order="F").result(),
            description="Reading tensorstore array",
        )
        if available_domain != requested_domain:
            # needs padding
            out = np.zeros(
                requested_domain.shape, dtype=array.dtype.numpy_dtype, order="F"
            )
            out[tuple(slice(0, data.shape[i]) for i in range(len(data.shape)))] = data
        else:
            out = data
        if not has_channel_dimension:
            out = np.expand_dims(out, 0)
        return out

    def resize(self, new_bbox: NDBoundingBox) -> None:
        array = self._array

        # Align with shards
        shard_shape = self.info.shard_shape
        new_bbox = new_bbox.with_bottomright_xyz(
            new_bbox.bottomright_xyz.ceildiv(shard_shape) * shard_shape
        )
        new_domain = tensorstore.IndexDomain(
            new_bbox.ndim + 1,
            shape=(self.info.num_channels,) + new_bbox.bottomright.to_tuple(),
            implicit_upper_bounds=tuple(True for _ in range(new_bbox.ndim + 1)),
            labels=array.domain.labels,
        )

        if new_domain != array.domain:
            # Check on-disk for changes to shape
            current_array = call_with_retries(
                lambda: tensorstore.open(
                    {
                        "driver": str(self.data_format),
                        "kvstore": self._make_kvstore(self._path),
                    },
                    context=TS_CONTEXT,
                ).result(),
                description="Opening tensorstore array for resizing",
            )
            if array.domain != current_array.domain:
                raise RuntimeError(
                    f"While resizing the Zarr array at {self._path}, a differing shape ({array.domain} != {current_array.domain}) was found in the currently persisted metadata."
                    + "This is likely happening because multiple processes changed the metadata of this array."
                )

            self._cached_array = call_with_retries(
                lambda: (
                    array.resize(
                        inclusive_min=None,
                        exclusive_max=new_domain.exclusive_max,
                        resize_metadata_only=True,
                    ).result()
                ),
                description="Resizing tensorstore array",
            )

    def write(self, bbox: NDBoundingBox, data: np.ndarray) -> None:
        if data.ndim == len(bbox):
            # the bbox does not include the channels, if data and bbox have the same size there is only 1 channel
            data = data.reshape((1,) + data.shape)

        assert data.ndim == len(bbox) + 1, (
            "The data has to have the same number of dimensions as the bounding box."
        )

        array = self._array

        requested_domain = tensorstore.IndexDomain(
            bbox.ndim + 1,
            inclusive_min=(0,) + bbox.topleft.to_tuple(),
            shape=(self.info.num_channels,) + bbox.size.to_tuple(),
        )
        call_with_retries(
            lambda: array[requested_domain].write(data).result(),
            description="Writing tensorstore array",
        )

    def _chunk_key_encoding(self) -> tuple[Literal["default", "v2"], Literal["/", "."]]:
        raise NotImplementedError

    def _list_bounding_boxes(
        self, kvstore: Any, shard_shape: Vec3Int, shape: VecInt
    ) -> Iterator[BoundingBox]:
        _type, separator = self._chunk_key_encoding()

        def _try_parse_ints(vec: Iterable[Any]) -> list[int] | None:
            output = []
            for value in vec:
                try:
                    output.append(int(value))
                except ValueError:  # noqa: PERF203
                    return None
            return output

        keys = call_with_retries(
            lambda: kvstore.list().result(), description="Listing keys in kvstore"
        )
        for key in keys:
            key_parts = key.decode("utf-8").split(separator)
            if _type == "default":
                if key_parts[0] != "c":
                    continue
                key_parts = key_parts[1:]
            if len(key_parts) != self._array.ndim:
                continue
            chunk_coords_list = _try_parse_ints(key_parts)
            if chunk_coords_list is None:
                continue

            if shape.axes[0] == "c":
                chunk_coords = Vec3Int(chunk_coords_list[1:])
            else:
                chunk_coords = Vec3Int(chunk_coords_list)

            yield BoundingBox(chunk_coords * shard_shape, shard_shape)

    def list_bounding_boxes(self) -> Iterator[BoundingBox]:
        kvstore = self._array.kvstore

        if kvstore.spec().to_json()["driver"] == "s3":
            raise NotImplementedError(
                "list_bounding_boxes() is not supported for s3 arrays."
            )

        _, _, shard_shape, _, shape = self._get_array_dimensions(self._array)

        if shape.axes != ("c", "x", "y", "z") and shape.axes != ("x", "y", "z"):
            raise NotImplementedError(
                "list_bounding_boxes() is not supported for non 3-D arrays."
            )

        # This needs to be a separate function because we need the NotImplementedError
        # to be raised immediately and not part of the iterator.
        return self._list_bounding_boxes(kvstore, shard_shape, shape)

    def close(self) -> None:
        if self._cached_array is not None:
            self._cached_array = None

    @property
    def _array(self) -> tensorstore.TensorStore:
        if self._cached_array is None:
            try:
                self._cached_array = call_with_retries(
                    lambda: tensorstore.open(
                        {
                            "driver": str(self.data_format),
                            "kvstore": self._make_kvstore(self._path),
                        },
                        context=TS_CONTEXT,
                    ).result(),
                    description="Creating tensorstore array",
                )
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

    def __getstate__(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        del d[""]
        return d

    def __setstate__(self, d: dict[str, Any]) -> None:
        d[""] = None
        self.__dict__ = d


class Zarr3Array(TensorStoreArray):
    data_format = DataFormat.Zarr3

    @classmethod
    def open(cls, path: Path) -> "Zarr3Array":
        return cls._open(UPath(path))

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
    def _has_compression_codecs(codecs: list[dict[str, str]]) -> bool:
        return any(c["name"] in ("blosc", "gzip", "zstd") for c in codecs)

    @classmethod
    def create(cls, path: Path, array_info: ArrayInfo) -> "Zarr3Array":
        assert array_info.data_format == cls.data_format
        upath = UPath(path)
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
                "kvstore": cls._make_kvstore(upath),
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
                                            "name": "zstd",
                                            "configuration": {
                                                "level": 5,
                                                "checksum": True,
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
            },
            context=TS_CONTEXT,
        ).result()
        return cls(upath, _array)

    def _chunk_key_encoding(self) -> tuple[Literal["default", "v2"], Literal["/", "."]]:
        metadata = self._array.spec().to_json()["metadata"]
        chunk_key_encoding = metadata["chunk_key_encoding"]
        _type = chunk_key_encoding["name"]
        separator = chunk_key_encoding.get("configuration", {}).get(
            "separator", "/" if _type == "default" else "."
        )
        assert _type in ["default", "v2"]
        assert separator in ["/", "."]
        return _type, separator


class Zarr2Array(TensorStoreArray):
    data_format = DataFormat.Zarr

    @classmethod
    def open(cls, path: Path) -> "Zarr2Array":
        return cls._open(UPath(path))

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
        assert array_info.chunks_per_shard == Vec3Int.full(1), (
            "Zarr (version 2) doesn't support sharding, use Zarr3 instead."
        )
        upath = UPath(path)
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
                "kvstore": cls._make_kvstore(upath),
                "metadata": {
                    "shape": shape,
                    "chunks": chunk_shape,
                    "dtype": array_info.voxel_type.str,
                    "fill_value": 0,
                    "order": "F",
                    "compressor": (
                        {
                            "id": "zstd",
                            "level": 5,
                        }
                        if array_info.compression_mode
                        else None
                    ),
                    "filters": None,
                    "dimension_separator": "/",
                },
                "create": True,
            },
            context=TS_CONTEXT,
        ).result()
        return cls(upath, _array)

    def _chunk_key_encoding(self) -> tuple[Literal["default", "v2"], Literal["/", "."]]:
        metadata = self._array.spec().to_json()["metadata"]
        separator = metadata.get("dimension_separator", ".")
        assert separator in ["/", "."]
        return "v2", separator

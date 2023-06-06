"""Utilities to work with the CLI of webknossos."""

import re
from collections import namedtuple
from enum import Enum
from os import environ, sep
from pathlib import Path
from types import TracebackType
from typing import Any, Generator, Iterator, Optional, Tuple, Type, Union

import numpy as np
from upath import UPath

from webknossos import BoundingBox, Mag

# constant for knossos dataset
CUBE_EDGE_LEN = 128
CUBE_SIZE = CUBE_EDGE_LEN**3
CUBE_SHAPE = (CUBE_EDGE_LEN,) * 3
KNOSSOS_CUBE_REGEX = re.compile(
    rf"x(\d+){re.escape(sep)}y(\d+){re.escape(sep)}z(\d+){re.escape(sep)}(.*\.raw)$"
)

KnossosDatasetInfo = namedtuple("KnossosDatasetInfo", ("dataset_path", "dtype"))
Vec2 = namedtuple("Vec2", ("x", "y"))
Vec3 = namedtuple("Vec3", ("x", "y", "z"))


class KnossosDataset:
    def __init__(self, root: Union[str, Path], dtype: np.dtype):
        self.root = Path(root)
        self.dtype = dtype

    def read(
        self, offset: Tuple[int, int, int], shape: Tuple[int, int, int]
    ) -> np.ndarray:
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert shape == CUBE_SHAPE
        return self.read_cube(tuple(x // CUBE_EDGE_LEN for x in offset))

    def write(self, offset: Tuple[int, int, int], data: np.ndarray) -> None:
        assert offset[0] % CUBE_EDGE_LEN == 0
        assert offset[1] % CUBE_EDGE_LEN == 0
        assert offset[2] % CUBE_EDGE_LEN == 0
        assert data.shape == CUBE_SHAPE
        self.write_cube(tuple(x // CUBE_EDGE_LEN for x in offset), data)

    def read_cube(self, cube_xyz: Tuple[int, ...]) -> np.ndarray:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            return np.zeros(CUBE_SHAPE, dtype=self.dtype)
        with open(filename, "rb") as cube_file:
            cube_data = np.fromfile(cube_file, dtype=self.dtype)
            if cube_data.size != CUBE_SIZE:
                padded_data = np.zeros(CUBE_SIZE, dtype=self.dtype)
                padded_data[0 : min(cube_data.size, CUBE_SIZE)] = cube_data[
                    0 : min(cube_data.size, CUBE_SIZE)
                ]
                cube_data = padded_data
            cube_data = cube_data.reshape(CUBE_SHAPE, order="F")
            return cube_data

    def write_cube(self, cube_xyz: Tuple[int, ...], cube_data: np.ndarray) -> None:
        filename = self.__get_only_raw_file_path(cube_xyz)
        if filename is None:
            filename = self.__get_cube_folder(cube_xyz) / self.__get_cube_file_name(
                cube_xyz
            )

        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as cube_file:
            cube_data.ravel(order="F").tofile(cube_file)

    def __get_cube_folder(self, cube_xyz: Tuple[int, ...]) -> Path:
        x, y, z = cube_xyz
        return (
            self.root / "x{:04d}".format(x) / "y{:04d}".format(y) / "z{:04d}".format(z)
        )

    def __get_cube_file_name(self, cube_xyz: Tuple[int, ...]) -> Path:
        x, y, z = cube_xyz
        return Path("cube_x{:04d}_y{:04d}_z{:04d}.raw".format(x, y, z))

    def __get_only_raw_file_path(self, cube_xyz: Tuple[int, ...]) -> Optional[Path]:
        cube_folder = self.__get_cube_folder(cube_xyz)
        raw_files = list(cube_folder.glob("*.raw"))
        assert len(raw_files) <= 1, "Found %d .raw files in %s" % (
            len(raw_files),
            cube_folder,
        )
        return raw_files[0] if len(raw_files) > 0 else None

    def list_files(self) -> Iterator[Path]:
        return self.root.glob("*/*/*/*.raw")

    def __parse_cube_file_name(self, filename: Path) -> Optional[Tuple[int, int, int]]:
        m = KNOSSOS_CUBE_REGEX.search(str(filename))
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    def list_cubes(self) -> Generator[Tuple[int, int, int], Any, None]:
        return (
            f
            for f in (self.__parse_cube_file_name(f) for f in self.list_files())
            if f is not None
        )

    def close(self) -> None:
        pass

    @staticmethod
    def open(root: Union[str, Path], dtype: np.dtype) -> "KnossosDataset":
        return KnossosDataset(root, dtype)

    def __enter__(self) -> "KnossosDataset":
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self.close()


class DistributionStrategy(str, Enum):
    """Enum of available distribution strategies.

    TODO: As soon as supported by typer this enum should be
        replaced with typing.Literal in type hint.
    """

    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    MULTIPROCESSING = "multiprocessing"
    DEBUGS_SEQUENTIAL = "debug_sequential"


class DataFormat(str, Enum):
    """Enum of available data formats."""

    WKW = "wkw"
    ZARR = "zarr"


class SamplingMode(str, Enum):
    """Enum of available sampling modes."""

    ANISOTROPIC = "anisotropic"
    ISOTROPIC = "isotropic"
    CONSTANT_Z = "constant_z"


class Order(str, Enum):
    """Enum of available orders."""

    C = "C"
    F = "F"


def parse_mag(mag_str: str) -> Mag:
    """Parses str input to Mag"""

    return Mag(mag_str)


def open_knossos(info: KnossosDatasetInfo) -> KnossosDataset:
    return KnossosDataset.open(info.dataset_path, np.dtype(info.dtype))


def parse_voxel_size(voxel_size_str: str) -> Tuple[float, float, float]:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(float(x) for x in voxel_size_str.split(","))
        if len(result) == 3:
            return Vec3(*result)
        raise ValueError(
            f"Expected three values formated like: 1.0,1.0,2.0 but got: {voxel_size_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize.\
Please format the voxel size like 1.0,1.0,2.0 ."
        ) from err


def parse_vec3int(vec3int_str: str) -> Tuple[int, int, int]:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(int(x) for x in vec3int_str.split(","))
        if len(result) == 1:
            return Vec3(*result * 3)
        if len(result) == 3:
            return Vec3(*result)
        raise ValueError(
            f"Expected three values formated like: 1,1,2 but got: {vec3int_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize.\
Please format the voxel size like 1,1,2 ."
        ) from err


def parse_vec2int(vec2int_str: str) -> Tuple[int, int]:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(int(x) for x in vec2int_str.split(","))
        if len(result) == 2:
            return result[0], result[1]
        raise ValueError(
            f"Expected three values formated like: 1,2 but got: {vec2int_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize.\
Please format the voxel size like 1,2 ."
        ) from err


def parse_bbox(bbox_str: str) -> BoundingBox:
    """Parses str input to BoundingBox."""

    try:
        result = tuple(int(x) for x in bbox_str.split(","))
        if len(result) == 6:
            return BoundingBox.from_tuple6(
                (result[0], result[1], result[2], result[3], result[4], result[5])
            )
        raise ValueError(
            f"Expected six values formated like: 0,0,0,5,5,5 but got: {bbox_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to BoundingBox.\
Please format the bounding box like 0,0,0,5,5,5 ."
        ) from err


def parse_path(value: str) -> UPath:
    """Parses a string value to a UPath."""

    if (
        (value.startswith("http://") or value.startswith("https://"))
        and "HTTP_BASIC_USER" in environ
        and "HTTP_BASIC_PASSWORD" in environ
    ):
        import aiohttp

        return UPath(
            value,
            client_kwargs={
                "auth": aiohttp.BasicAuth(
                    environ["HTTP_BASIC_USER"], environ["HTTP_BASIC_PASSWORD"]
                )
            },
        )
    if (
        (value.startswith("webdav+http://") or value.startswith("webdav+https://"))
        and "HTTP_BASIC_USER" in environ
        and "HTTP_BASIC_PASSWORD" in environ
    ):
        return UPath(
            value,
            auth=(environ["HTTP_BASIC_USER"], environ["HTTP_BASIC_PASSWORD"]),
        )
    if value.startswith("s3://") and "S3_ENDPOINT_URL" in environ:
        return UPath(
            value,
            client_kwargs={"endpoint_url": environ["S3_ENDPOINT_URL"]},
        )

    return UPath(value)


def pad_or_crop_to_size_and_topleft(
    cube_data: np.ndarray, target_size: np.ndarray, target_topleft: np.ndarray
) -> np.ndarray:
    """
    Given an numpy array and a target_size/target_topleft, the array
    will be padded so that it is within the bounding box descriped by topleft and size.
    If the input data is too large, the data will be cropped (evenly from opposite sides
    with the assumption that the most important data is in the center).
    """

    # Pad to size
    half_padding = (target_size - cube_data.shape) / 2
    half_padding = np.clip(half_padding, 0, None)
    left_padding = np.floor(half_padding).astype(np.uint32)
    right_padding = np.floor(half_padding).astype(np.uint32)

    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (left_padding[1], right_padding[1]),
            (left_padding[2], right_padding[2]),
            (0, 0),
        ),
    )

    # Potentially crop to size
    half_overflow = (cube_data.shape - target_size) / 2
    half_overflow = np.clip(half_overflow, 0, None)
    left_overflow = np.floor(half_overflow).astype(np.uint32)
    right_overflow = np.floor(half_overflow).astype(np.uint32)
    cube_data = cube_data[
        :,
        left_overflow[1] : cube_data.shape[1] - right_overflow[1],
        left_overflow[2] : cube_data.shape[2] - right_overflow[2],
        :,
    ]

    # Pad to topleft
    cube_data = np.pad(
        cube_data,
        (
            (0, 0),
            (target_topleft[1], max(0, target_size[1] - cube_data.shape[1])),
            (target_topleft[2], max(0, target_size[2] - cube_data.shape[2])),
            (target_topleft[3], max(0, target_size[3] - cube_data.shape[3])),
        ),
    )

    return cube_data

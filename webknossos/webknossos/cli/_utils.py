"""Utilities to work with the CLI of webknossos."""

from collections import namedtuple
from enum import Enum
from os import environ
from typing import Tuple

from upath import UPath

from webknossos import BoundingBox, Mag


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


Vec2 = namedtuple("Vec2", ("x", "y"))
Vec3 = namedtuple("Vec3", ("x", "y", "z"))


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

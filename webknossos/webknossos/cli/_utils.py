"""Utilities to work with the CLI of webknossos."""

from collections import namedtuple
from enum import Enum
from os import environ
from typing import Tuple, Union

import numpy as np
from upath import UPath

from ..geometry import BoundingBox, Mag, Vec3Int

VoxelSize = namedtuple("VoxelSize", ("x", "y", "z"))
Vec2Int = namedtuple("Vec2Int", ("x", "y"))


class DistributionStrategy(str, Enum):
    """Enum of available distribution strategies.

    TODO  pylint: disable=fixme
    - As soon as supported by typer this enum should be
    replaced with typing.Literal in type hint.
    """

    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    MULTIPROCESSING = "multiprocessing"
    DEBUG_SEQUENTIAL = "debug_sequential"


class LayerCategory(str, Enum):
    """Enum of available layer categories.

    TODO  pylint: disable=fixme
    - As soon as supported by typer this enum should be
    replaced with typing.Literal in type hint.
    """

    COLOR = "color"
    SEGMENTATION = "segmentation"


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


def parse_voxel_size(voxel_size_str: str) -> Tuple[float, float, float]:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(float(x) for x in voxel_size_str.split(","))
        if len(result) == 3:
            return VoxelSize(*result)
        raise ValueError(
            f"Expected three values formated like: 1.0,1.0,2.0 but got: {voxel_size_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize. "
            "Please format the voxel size like 1.0,1.0,2.0 ."
        ) from err


def parse_vec3int(vec3int_like: Union[str, Vec3Int]) -> Vec3Int:
    """Parses str input to tuple of three integers."""
    try:
        if isinstance(vec3int_like, Vec3Int):
            return vec3int_like
        result = tuple(int(x) for x in vec3int_like.split(","))
        if len(result) == 1:
            return Vec3Int(result * 3)
        if len(result) == 3:
            return Vec3Int(result)
        raise ValueError(
            f"Expected three values formated like: 1,1,2 but got: {vec3int_like}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize. "
            "Please format the voxel size like 1,1,2 ."
        ) from err


def parse_vec2int(vec2int_str: str) -> Vec2Int:
    """Parses str input to tuple of two integers."""
    try:
        result = tuple(int(x) for x in vec2int_str.split(","))
        if len(result) == 2:
            return Vec2Int(*result)
        raise ValueError(
            f"Expected three values formated like: 1,2 but got: {vec2int_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize. "
            "Please format the voxel size like 1,2 ."
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

"""Utilities to work with the CLI of webknossos."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from os import environ
from typing import NamedTuple
from urllib.parse import urlparse

import numpy as np
from upath import UPath

from ..annotation.annotation import _ANNOTATION_URL_REGEX, Annotation
from ..client import webknossos_context
from ..client._resolve_short_link import resolve_short_link
from ..dataset import Dataset, RemoteDataset
from ..dataset.abstract_dataset import _DATASET_DEPRECATED_URL_REGEX, _DATASET_URL_REGEX
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import is_fs_path


class VoxelSizeTuple(NamedTuple):
    x: float
    y: float
    z: float


class RescaleValues(NamedTuple):
    min: float
    max: float


class Vec2Int(NamedTuple):
    x: int
    y: int


class DistributionStrategy(str, Enum):
    """Enum of available distribution strategies.

    TODO
    - As soon as supported by typer this enum should be
    replaced with typing.Literal in type hint.
    https://github.com/tiangolo/typer/pull/669
    """

    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    MULTIPROCESSING = "multiprocessing"
    SEQUENTIAL = "sequential"


class LayerCategory(str, Enum):
    """Enum of available layer categories.

    TODO
    - As soon as supported by typer this enum should be
    replaced with typing.Literal in type hint.
    https://github.com/tiangolo/typer/pull/669
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


def parse_voxel_size(voxel_size_str: str) -> tuple[float, float, float]:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(float(x) for x in voxel_size_str.split(","))
        if len(result) == 3:
            return VoxelSizeTuple(*result)
        raise ValueError(
            f"Expected three values formatted like: 1.0,1.0,2.0 but got: {voxel_size_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize. "
            "Please format the voxel size like 1.0,1.0,2.0 ."
        ) from err


def parse_rescale_values(rescale_str: str) -> RescaleValues:
    """Parses str input to tuple of two floats."""
    try:
        result = tuple(float(x) for x in rescale_str.split(","))
        if len(result) == 2:
            return RescaleValues(*result)
        raise ValueError(
            f"Expected two values formatted like: 1.0,2.0 but got: {rescale_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to RescaleValues. "
            "Please format the voxel size like 1.0,2.0 ."
        ) from err


def parse_vec3int(vec3int_like: str | Vec3Int) -> Vec3Int:
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
            f"Expected three values formatted like: 1,1,2 but got: {vec3int_like}"
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
            f"Expected three values formatted like: 1,2 but got: {vec2int_str}"
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
            f"Expected six values formatted like: 0,0,0,5,5,5 but got: {bbox_str}"
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
            endpoint_url=environ["S3_ENDPOINT_URL"],
        )

    return UPath(value)


def pad_or_crop_to_size_and_topleft(
    cube_data: np.ndarray, target_size: np.ndarray, target_topleft: np.ndarray
) -> np.ndarray:
    """
    Given an numpy array and a target_size/target_topleft, the array
    will be padded so that it is within the bounding box described by topleft and size.
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


def prepare_shard_shape(
    *,
    chunk_shape: Vec3Int | None,
    shard_shape: Vec3Int | None,
    chunks_per_shard: Vec3Int | None,
) -> Vec3Int | None:
    if chunk_shape is None:
        chunk_shape = DEFAULT_CHUNK_SHAPE

    if shard_shape is not None and chunks_per_shard is not None:
        raise ValueError(
            "shard_shape and chunks_per_shard must not be specified at the same time."
        )

    if shard_shape is None:
        if chunks_per_shard is None:
            return None
        return chunk_shape * chunks_per_shard
    else:
        if shard_shape % chunk_shape != Vec3Int.zeros():
            raise ValueError(
                f"The shard_shape {shard_shape} must be cleanly divisible by the chunk_shape {chunk_shape}."
            )
        return shard_shape


@contextmanager
def open_dataset(
    source: UPath,
    annotation_ok: bool,
    token: str | None = None,
) -> Iterator[Dataset | RemoteDataset]:
    if not is_fs_path(source):
        url = resolve_short_link(str(source))
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        with webknossos_context(url=domain, token=token):
            if re.match(_DATASET_URL_REGEX, url) or re.match(
                _DATASET_DEPRECATED_URL_REGEX, url
            ):
                yield RemoteDataset.open(url)
            elif re.match(_ANNOTATION_URL_REGEX, url):
                if not annotation_ok:
                    raise ValueError(
                        "The provided URL leads to an annotation, not a dataset."
                    )
                yield Annotation.open_as_remote_dataset(annotation_id_or_url=url)
            else:
                raise ValueError(
                    "The provided URL does not lead to a WEBKNOSSOS dataset or annotation."
                )
    else:
        yield Dataset.open(source)

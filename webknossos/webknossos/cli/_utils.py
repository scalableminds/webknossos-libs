"""Utilities to work with the CLI of webknossos."""

import json
import logging
import re
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from functools import lru_cache
from multiprocessing import cpu_count
from os import environ
from typing import Annotated, NamedTuple
from urllib.parse import urlparse

import numpy as np
import typer
from cluster_tools import BatchingExecutor, Executor, get_executor
from upath import UPath

from ..annotation.annotation import _ANNOTATION_URL_REGEX, Annotation
from ..client import webknossos_context
from ..client._resolve_short_link import resolve_short_link
from ..dataset import Dataset, RemoteDataset
from ..dataset.abstract_dataset import _DATASET_DEPRECATED_URL_REGEX, _DATASET_URL_REGEX
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE, DEFAULT_DATA_FORMAT
from ..dataset.remote_dataset import RemoteAccessMode
from ..dataset_properties import DataFormat
from ..geometry import BoundingBox, Mag, Vec3Int
from ..utils import is_fs_path, set_s3fs_retry_settings

logger = logging.getLogger(__name__)


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
    """Enum of available distribution strategies."""

    SLURM = "slurm"
    SLURM_BATCHING = "slurm+batching"
    KUBERNETES = "kubernetes"
    MULTIPROCESSING = "multiprocessing"
    SEQUENTIAL = "sequential"


DEFAULT_JOBS: int = cpu_count()

JobsOption = Annotated[
    int,
    typer.Option(
        help="Number of processes to be spawned.",
        rich_help_panel="Executor options",
    ),
]

DistributionStrategyOption = Annotated[
    DistributionStrategy,
    typer.Option(
        help="Strategy to distribute the task across CPUs or nodes.",
        rich_help_panel="Executor options",
    ),
]

JobResourcesOption = Annotated[
    str | None,
    typer.Option(
        help="Necessary when using slurm as distribution strategy. Should be a JSON string "
        '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
        rich_help_panel="Executor options",
    ),
]


def get_executor_for_args(
    *,
    jobs: int,
    distribution_strategy: DistributionStrategy,
    job_resources: str | None,
) -> AbstractContextManager[Executor]:
    if distribution_strategy == DistributionStrategy.MULTIPROCESSING:
        logger.info(f"Using pool of {jobs} workers.")
        return get_executor("multiprocessing", max_workers=jobs)
    if distribution_strategy in (
        DistributionStrategy.SLURM,
        DistributionStrategy.SLURM_BATCHING,
        DistributionStrategy.KUBERNETES,
    ):
        if job_resources is None:
            resources_example = (
                '{"mem": "32G"}'
                if distribution_strategy != DistributionStrategy.KUBERNETES
                else '{"memory": "32G"}'
            )
            raise typer.BadParameter(
                f"Job resources has to be provided when using {distribution_strategy.value} as distribution strategy. "
                f"Example: --job-resources='{resources_example}'",
                param_hint="--job-resources",
            )
        job_resources_parsed = json.loads(job_resources)

        if distribution_strategy == DistributionStrategy.SLURM_BATCHING:
            target_job_count = job_resources_parsed.get("target_job_count")
            batch_size = job_resources_parsed.get("batch_size")
            if target_job_count is None and batch_size is None:
                resources_example = '{"target_job_count": 100, "mem": "32G"}'
                raise typer.BadParameter(
                    f"target_job_count or batch_size have to be provided when using {distribution_strategy.value} as distribution strategy. "
                    f"Example: --job-resources='{resources_example}'",
                    param_hint="--job-resources",
                )
            if target_job_count is not None and batch_size is not None:
                raise typer.BadParameter(
                    f"target_job_count and batch_size can not be provided at the same time when using {distribution_strategy.value} as distribution strategy.",
                    param_hint="--job-resources",
                )
            distribution_strategy = DistributionStrategy.SLURM
            logger.info(f"Using {distribution_strategy.value} cluster with batching.")
            return BatchingExecutor(
                get_executor(
                    distribution_strategy.value,
                    debug=True,
                    keep_logs=True,
                    job_resources=job_resources_parsed,
                ),
                target_job_count=target_job_count,
                batch_size=batch_size,
            )

        logger.info(f"Using {distribution_strategy.value} cluster.")
        return get_executor(
            distribution_strategy.value,
            debug=True,
            keep_logs=True,
            job_resources=job_resources_parsed,
        )

    if distribution_strategy == DistributionStrategy.SEQUENTIAL:
        return get_executor(
            distribution_strategy.value,
            debug=True,
            keep_logs=True,
        )

    raise typer.BadParameter(
        f"Unknown distribution strategy: {distribution_strategy.value}",
        param_hint="--distribution-strategy",
    )


class LayerCategory(str, Enum):
    """Enum of available layer categories."""

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


@lru_cache(maxsize=1)
def _set_s3fs_retry_settings() -> None:
    set_s3fs_retry_settings()


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


DEFAULT_DATA_FORMAT_STR: str = str(DEFAULT_DATA_FORMAT)

DataFormatOption = Annotated[
    DataFormat,
    typer.Option(help="Data format to store the target dataset in."),
]

ChunkShapeOption = Annotated[
    Vec3Int,
    typer.Option(
        help="Number of voxels to be stored as a chunk in the output format "
        "(e.g. `32` or `32,32,32`).",
        parser=parse_vec3int,
        metavar="Vec3Int",
    ),
]

ShardShapeOption = Annotated[
    Vec3Int | None,
    typer.Option(
        help="Number of voxels to be stored as a shard in the output format "
        "(e.g. `1024` or `1024,1024,1024`).",
        parser=parse_vec3int,
        metavar="Vec3Int",
    ),
]

ChunksPerShardOption = Annotated[
    Vec3Int | None,
    typer.Option(
        help="Deprecated, use --shard-shape. Number of chunks to be stored as a shard in the output format "
        "(e.g. `32` or `32,32,32`).",
        parser=parse_vec3int,
        metavar="Vec3Int",
    ),
]

ExistsOkOption = Annotated[
    bool,
    typer.Option(help="Whether it should overwrite an existing dataset."),
]


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
    if value.startswith("s3://"):
        _set_s3fs_retry_settings()

        if "S3_ENDPOINT_URL" in environ:
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
    access_mode: RemoteAccessMode | None = None,
) -> Iterator[Dataset | RemoteDataset]:
    if not is_fs_path(source):
        url = resolve_short_link(str(source))
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        with webknossos_context(url=domain, token=token):
            if re.match(_DATASET_URL_REGEX, url) or re.match(
                _DATASET_DEPRECATED_URL_REGEX, url
            ):
                yield RemoteDataset.open(url, access_mode=access_mode)
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

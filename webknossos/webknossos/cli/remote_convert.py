"""This module converts an image stack, uploads it to WEBKNOSSOS, and returns a RemoteDataset."""

import tempfile
from argparse import Namespace
from multiprocessing import cpu_count
from typing import Annotated, Any

import typer
from upath import UPath

from ..client._defaults import DEFAULT_WEBKNOSSOS_URL
from ..client.context import webknossos_context
from ..dataset import Dataset, RemoteFolder, SamplingModes
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE, DEFAULT_DATA_FORMAT
from ..dataset_properties import DataFormat, LengthUnit, VoxelSize
from ..dataset_properties.structuring import DEFAULT_LENGTH_UNIT_STR
from ..geometry import Mag, Vec3Int
from ..utils import get_executor_for_args
from ._utils import (
    DistributionStrategy,
    LayerCategory,
    SamplingMode,
    VoxelSizeTuple,
    parse_mag,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
    prepare_shard_shape,
)


def main(
    *,
    source: Annotated[
        Any,
        typer.Argument(
            help="Path to your image data.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    voxel_size: Annotated[
        VoxelSizeTuple,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma-separated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VoxelSize",
            show_default=False,
        ),
    ],
    unit: Annotated[
        LengthUnit,
        typer.Option(
            help="The unit of the voxel size.",
        ),
    ] = DEFAULT_LENGTH_UNIT_STR,  # type:ignore
    webknossos_url: Annotated[
        str,
        typer.Option(
            help="URL to WEBKNOSSOS instance.",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_URL",
        ),
    ] = DEFAULT_WEBKNOSSOS_URL,
    token: Annotated[
        str | None,
        typer.Option(
            help="Authentication token for WEBKNOSSOS instance "
            "(https://webknossos.org/auth/token).",
            rich_help_panel="WEBKNOSSOS context",
            envvar="WK_TOKEN",
        ),
    ] = None,
    dataset_name: Annotated[
        str | None,
        typer.Option(
            help="Name for the dataset on WEBKNOSSOS. "
            "If not provided, the source directory name is used.",
        ),
    ] = None,
    folder: Annotated[
        str | None,
        typer.Option(
            help="WEBKNOSSOS dataset folder in which the dataset should be placed. "
            "Specify the folder path as a string, separated by `/`. "
            "Example: `Datasets/mySubfolder`. "
            "If not provided, the root folder is used.",
        ),
    ] = None,
    layer_name: Annotated[
        str | None,
        typer.Option(
            help="This name is used if only one layer is created. Otherwise this name is used as a common prefix for all layers.",
        ),
    ] = None,
    category: Annotated[
        LayerCategory | None,
        typer.Option(
            help="The category of the layer that should be created.",
        ),
    ] = None,
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = str(DEFAULT_DATA_FORMAT),  # type:ignore
    chunk_shape: Annotated[
        Vec3Int,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    shard_shape: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Number of voxels to be stored as a shard in the output format "
            "(e.g. `1024` or `1024,1024,1024`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    chunks_per_shard: Annotated[
        Vec3Int | None,
        typer.Option(
            help="Deprecated, use --shard-shape. Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = None,
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
    ] = True,
    downsample: Annotated[
        bool, typer.Option(help="Downsample the dataset locally before uploading.")
    ] = True,
    max_mag: Annotated[
        Mag | None,
        typer.Option(
            help="Create downsampled magnifications up to the magnification specified by this argument. "
            "If omitted, the coarsest magnification will be determined by using the bounding box of the layer. "
            "Should be number or hyphen-separated string (e.g. `2` or `2-2-2`).",
            parser=parse_mag,
        ),
    ] = None,
    interpolation_mode: Annotated[
        str,
        typer.Option(
            help="The interpolation mode that should be used "
            "(median, mode, nearest, bilinear or bicubic)."
        ),
    ] = "default",
    sampling_mode: Annotated[
        SamplingMode, typer.Option(help="The sampling mode to use.")
    ] = SamplingMode.ANISOTROPIC,
    batch_size: Annotated[
        int | None,
        typer.Option(
            help="Number of images to be processed in one batch (influences RAM consumption). "
            "When creating a WKW dataset, batch-size must be a multiple of chunk-shape's z dimension. "
            "When converting to Zarr, batch-size must be a multiple of the z dimension of the "
            "shard shape (chunk-shape x chunks-per-shard).",
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of processes to be spawned.",
            rich_help_panel="Executor options",
        ),
    ] = cpu_count(),
    distribution_strategy: Annotated[
        DistributionStrategy,
        typer.Option(
            help="Strategy to distribute the task across CPUs or nodes.",
            rich_help_panel="Executor options",
        ),
    ] = DistributionStrategy.MULTIPROCESSING,
    job_resources: Annotated[
        str | None,
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job-resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Convert an image stack to a WEBKNOSSOS dataset and upload it directly."""

    shard_shape = prepare_shard_shape(
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        chunks_per_shard=chunks_per_shard,
    )

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy.value,
        job_resources=job_resources,
    )
    voxel_size_with_unit = VoxelSize(voxel_size, unit)
    mode = SamplingModes.parse(sampling_mode.value)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with get_executor_for_args(args=executor_args) as executor:
            dataset = Dataset.from_images(
                source,
                UPath(tmp_dir) / "dataset",
                name=dataset_name,
                voxel_size_with_unit=voxel_size_with_unit,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                data_format=data_format,
                executor=executor,
                compress=compress,
                layer_name=layer_name,
                batch_size=batch_size,
                layer_category=category.value if category else None,
            )
        if downsample:
            with get_executor_for_args(args=executor_args) as executor:
                dataset.downsample(
                    coarsest_mag=max_mag,
                    interpolation_mode=interpolation_mode,
                    compress=compress,
                    sampling_mode=mode,
                    executor=executor,
                )

        with webknossos_context(url=webknossos_url, token=token):
            folder_obj: None | RemoteFolder = None
            if folder is not None:
                folder_obj = RemoteFolder.get_by_path(folder)
            dataset.upload(
                new_dataset_name=dataset_name,
                folder=folder_obj,
            )

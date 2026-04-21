"""This module converts an image stack to a WEBKNOSSOS dataset."""

import tempfile
from typing import Annotated, Any

import typer
from upath import UPath

from ..client._defaults import DEFAULT_WEBKNOSSOS_URL
from ..client.context import webknossos_context
from ..dataset import Dataset, RemoteFolder, SamplingModes, TransferMode
from ..dataset.defaults import DEFAULT_CHUNK_SHAPE
from ..dataset_properties import LengthUnit, VoxelSize
from ..dataset_properties.structuring import DEFAULT_LENGTH_UNIT_STR
from ..geometry import Mag
from ..utils import rmtree
from ._utils import (
    DEFAULT_DATA_FORMAT_STR,
    DEFAULT_JOBS,
    ChunkShapeOption,
    ChunksPerShardOption,
    DataFormatOption,
    DistributionStrategy,
    DistributionStrategyOption,
    JobResourcesOption,
    JobsOption,
    LayerCategory,
    SamplingMode,
    ShardShapeOption,
    VoxelSizeTuple,
    get_executor_for_args,
    parse_mag,
    parse_path,
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
    target: Annotated[
        Any,
        typer.Argument(
            help="Target path to save your WEBKNOSSOS dataset. Required unless --upload is set.",
            show_default=False,
            parser=parse_path,
        ),
    ] = None,
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
    data_format: DataFormatOption = DEFAULT_DATA_FORMAT_STR,  # type: ignore
    name: Annotated[
        str | None,
        typer.Option(
            help="Name for the WEBKNOSSOS dataset. "
            "If not provided, the final component of the target path is used (local) "
            "or the source directory name (upload)."
        ),
    ] = None,
    chunk_shape: ChunkShapeOption = DEFAULT_CHUNK_SHAPE,
    shard_shape: ShardShapeOption = None,
    chunks_per_shard: ChunksPerShardOption = None,
    compress: Annotated[
        bool, typer.Option(help="Enable compression of the target dataset.")
    ] = True,
    downsample: Annotated[
        bool, typer.Option(help="Downsample the target dataset.")
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
    overwrite_existing: Annotated[
        bool,
        typer.Option(
            help="Clear target folder if it already exists. Not enabled by default. Use with caution.",
            show_default=False,
        ),
    ] = False,
    upload: Annotated[
        bool,
        typer.Option(
            help="Convert to a temporary directory and upload the result to a WEBKNOSSOS server. "
            "Requires --token (or WK_TOKEN). TARGET must not be provided.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = False,
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
    folder: Annotated[
        str | None,
        typer.Option(
            help="WEBKNOSSOS dataset folder in which the dataset should be placed. "
            "Specify the folder path as a string, separated by `/`. "
            "Example: `Datasets/mySubfolder`. "
            "If not provided, the root folder is used.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = None,
    transfer_mode: Annotated[
        TransferMode,
        typer.Option(
            help="The transfer mode to use when uploading. 'http' is the default. "
            "Other modes like 'copy', 'move+symlink', 'symlink' are for users with direct filesystem access to the WEBKNOSSOS datastore.",
            rich_help_panel="WEBKNOSSOS context",
        ),
    ] = TransferMode.HTTP,
    jobs: JobsOption = DEFAULT_JOBS,
    distribution_strategy: DistributionStrategyOption = DistributionStrategy.MULTIPROCESSING,
    job_resources: JobResourcesOption = None,
) -> None:
    """Automatic detection of an image stack and conversion to a WEBKNOSSOS dataset."""

    if upload and target is not None:
        raise typer.BadParameter("TARGET must not be provided when --upload is set.")
    if not upload and target is None:
        raise typer.BadParameter("TARGET is required unless --upload is set.")

    shard_shape = prepare_shard_shape(
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        chunks_per_shard=chunks_per_shard,
    )

    voxel_size_with_unit = VoxelSize(voxel_size, unit)
    mode = SamplingModes.parse(sampling_mode.value)

    def _convert_and_downsample(target_path: UPath) -> Dataset:
        with get_executor_for_args(
            jobs=jobs,
            distribution_strategy=distribution_strategy,
            job_resources=job_resources,
        ) as executor:
            dataset = Dataset.from_images(
                source,
                target_path,
                name=name,
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
            with get_executor_for_args(
                jobs=jobs,
                distribution_strategy=distribution_strategy,
                job_resources=job_resources,
            ) as executor:
                dataset.downsample(
                    coarsest_mag=max_mag,
                    interpolation_mode=interpolation_mode,
                    compress=compress,
                    sampling_mode=mode,
                    executor=executor,
                )
        return dataset

    if upload:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = _convert_and_downsample(UPath(tmp_dir) / "dataset")
            with webknossos_context(url=webknossos_url, token=token):
                folder_obj: None | RemoteFolder = None
                if folder is not None:
                    folder_obj = RemoteFolder.get_by_path(folder)
                remote_dataset = dataset.upload(
                    new_dataset_name=name,
                    folder=folder_obj,
                    transfer_mode=transfer_mode,
                )
                print(f"Uploaded to: {remote_dataset.url}")
    else:
        if overwrite_existing and target.exists():
            rmtree(target)
        _convert_and_downsample(target)

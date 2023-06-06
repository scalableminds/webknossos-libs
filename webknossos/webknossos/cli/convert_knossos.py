"""This module converts a KNOSSOS dataset into a WEBKNOSSOS dataset."""

import logging
from argparse import Namespace
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional, Tuple, cast

import numpy as np
import typer
from typing_extensions import Annotated

from webknossos import COLOR_CATEGORY, BoundingBox, Dataset, Mag, Vec3Int, View
from webknossos.cli._utils import (
    CUBE_EDGE_LEN,
    DataFormat,
    DistributionStrategy,
    KnossosDatasetInfo,
    open_knossos,
    parse_mag,
    parse_path,
    parse_vec3int,
    parse_voxel_size,
)
from webknossos.dataset.defaults import DEFAULT_CHUNK_SHAPE, DEFAULT_CHUNKS_PER_SHARD
from webknossos.utils import get_executor_for_args, time_start, time_stop


def convert_cube_job(
    source_knossos_info: KnossosDatasetInfo, args: Tuple[View, int]
) -> None:
    target_view, _ = args

    time_start(f"Converting of {target_view.bounding_box}")
    cube_size = cast(Tuple[int, int, int], (CUBE_EDGE_LEN,) * 3)

    offset = target_view.bounding_box.in_mag(target_view.mag).topleft
    size = target_view.bounding_box.in_mag(target_view.mag).size
    buffer = np.zeros(size.to_tuple(), dtype=target_view.get_dtype())
    with open_knossos(source_knossos_info) as source_knossos:
        for x in range(0, size.x, CUBE_EDGE_LEN):
            for y in range(0, size.y, CUBE_EDGE_LEN):
                for z in range(0, size.z, CUBE_EDGE_LEN):
                    cube_data = source_knossos.read(
                        (offset + Vec3Int(x, y, z)).to_tuple(), cube_size
                    )
                    buffer[
                        x : (x + CUBE_EDGE_LEN),
                        y : (y + CUBE_EDGE_LEN),
                        z : (z + CUBE_EDGE_LEN),
                    ] = cube_data
    target_view.write(buffer)

    time_stop(f"Converting of {target_view.bounding_box}")


def convert_knossos(
    source_path: Path,
    target_path: Path,
    layer_name: str,
    dtype: str,
    voxel_size: Tuple[float, float, float],
    data_format: DataFormat,
    chunk_shape: Vec3Int,  # in target-mag
    chunks_per_shard: Vec3Int,
    mag: Mag = Mag(1),
    args: Optional[Namespace] = None,
) -> None:
    """Performs the conversion of a KNOSSOS dataset to a WEBKNOSSOS dataset."""

    source_knossos_info = KnossosDatasetInfo(source_path, dtype)

    target_dataset = Dataset(target_path, voxel_size, exist_ok=True)
    target_layer = target_dataset.get_or_add_layer(
        layer_name,
        COLOR_CATEGORY,
        data_format=data_format,
        dtype_per_channel=dtype,
    )

    with open_knossos(source_knossos_info) as source_knossos:
        knossos_cubes = np.array(list(source_knossos.list_cubes()))
        if len(knossos_cubes) == 0:
            logging.error(
                "No input KNOSSOS cubes found. Make sure to pass the path which "
                "points to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
            )
            exit(1)

        min_xyz = knossos_cubes.min(axis=0) * CUBE_EDGE_LEN
        max_xyz = (knossos_cubes.max(axis=0) + 1) * CUBE_EDGE_LEN
        target_layer.bounding_box = BoundingBox(
            Vec3Int(min_xyz), Vec3Int(max_xyz - min_xyz)
        )

    target_mag = target_layer.get_or_add_mag(
        mag, chunk_shape=chunk_shape, chunks_per_shard=chunks_per_shard
    )

    with get_executor_for_args(args) as executor:
        target_mag.for_each_chunk(
            partial(convert_cube_job, source_knossos_info),
            chunk_shape=chunk_shape * mag * chunks_per_shard,
            executor=executor,
            progress_desc=f"Converting knossos layer {layer_name}",
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
            help="Target path to save your WEBKNOSSOS dataset.",
            show_default=False,
            parser=parse_path,
        ),
    ],
    layer_name: Annotated[
        str,
        typer.Option(help="Name of the cubed layer (color or segmentation)"),
    ] = "color",
    voxel_size: Annotated[
        Any,
        typer.Option(
            help="The size of one voxel in source data in nanometers. "
            "Should be a comma seperated string (e.g. 11.0,11.0,20.0).",
            parser=parse_voxel_size,
            metavar="VOXEL_SIZE",
        ),
    ],
    dtype: Annotated[
        str, typer.Option(help="Target datatype (e.g. uint8, uint16, uint32)")
    ] = "uint8",
    data_format: Annotated[
        DataFormat,
        typer.Option(
            help="Data format to store the target dataset in.",
        ),
    ] = DataFormat.WKW,
    chunk_shape: Annotated[
        Any,
        typer.Option(
            help="Number of voxels to be stored as a chunk in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNK_SHAPE,
    chunks_per_shard: Annotated[
        Any,
        typer.Option(
            help="Number of chunks to be stored as a shard in the output format "
            "(e.g. `32` or `32,32,32`).",
            parser=parse_vec3int,
            metavar="Vec3Int",
        ),
    ] = DEFAULT_CHUNKS_PER_SHARD,
    mag: Annotated[
        Mag,
        typer.Option(
            help="Mag to start upsampling from. "
            "Should be number or minus seperated string (e.g. 2 or 2-2-2).",
            parser=parse_mag,
            metavar="MAG",
        ),
    ] = 1,  # type: ignore
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
        Optional[str],
        typer.Option(
            help="Necessary when using slurm as distribution strategy. Should be a JSON string "
            '(e.g., --job_resources=\'{"mem": "10M"}\')\'',
            rich_help_panel="Executor options",
        ),
    ] = None,
) -> None:
    """Convert your KNOSSOS dataset to a WEBKNOSOOS dataset."""

    executor_args = Namespace(
        jobs=jobs,
        distribution_strategy=distribution_strategy,
        job_resources=job_resources,
    )

    convert_knossos(
        source,
        target,
        layer_name,
        dtype,
        voxel_size,
        data_format,
        chunk_shape,
        chunks_per_shard,
        mag,
        executor_args,
    )

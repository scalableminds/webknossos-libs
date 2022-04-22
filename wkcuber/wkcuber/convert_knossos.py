import logging
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, cast

import numpy as np
from webknossos import COLOR_CATEGORY, BoundingBox, DataFormat, Dataset, Vec3Int, View
from webknossos.utils import time_start, time_stop

from ._internal.knossos import CUBE_EDGE_LEN
from ._internal.utils import (
    KnossosDatasetInfo,
    add_data_format_flags,
    add_distribution_flags,
    add_voxel_size_flag,
    add_verbose_flag,
    get_executor_for_args,
    open_knossos,
    parse_path,
    setup_logging,
    setup_warnings,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path",
        help="Directory containing the source KNOSSOS dataset.",
        type=Path,
    )

    parser.add_argument(
        "target_path",
        help="Output directory for the generated WKW dataset.",
        type=parse_path,
    )

    parser.add_argument(
        "--layer_name",
        "-l",
        help="Name of the cubed layer (color or segmentation)",
        default="color",
    )

    parser.add_argument(
        "--dtype",
        "-d",
        help="Target datatype (e.g. uint8, uint16, uint32)",
        default="uint8",
    )

    add_voxel_size_flag(parser)

    parser.add_argument("--mag", "-m", help="Magnification level", type=int, default=1)

    add_verbose_flag(parser)
    add_distribution_flags(parser)
    add_data_format_flags(parser)

    return parser


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
                        y : (y + CUBE_EDGE_LEN),
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
    chunk_size: Vec3Int,
    chunks_per_shard: Vec3Int,
    mag: int = 1,
    args: Optional[Namespace] = None,
) -> None:
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
                "No input KNOSSOS cubes found. Make sure to pass the path which points to a KNOSSOS magnification (e.g., testdata/knossos/color/1)."
            )
            exit(1)

        min_xyz = knossos_cubes.min(axis=0) * CUBE_EDGE_LEN
        max_xyz = (knossos_cubes.max(axis=0) + 1) * CUBE_EDGE_LEN
        target_layer.bounding_box = BoundingBox(
            Vec3Int(min_xyz), Vec3Int(max_xyz - min_xyz)
        )

    target_mag = target_layer.get_or_add_mag(
        mag, chunk_size=chunk_size, chunks_per_shard=chunks_per_shard
    )

    with get_executor_for_args(args) as executor:
        target_mag.for_each_chunk(
            partial(convert_cube_job, source_knossos_info),
            chunk_size=chunk_size * chunks_per_shard,
            executor=executor,
            progress_desc=f"Converting knossos layer {layer_name}",
        )


def main(args: Namespace) -> None:
    convert_knossos(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.voxel_size,
        args.data_format,
        args.chunk_size,
        args.chunks_per_shard,
        args.mag,
        args,
    )


if __name__ == "__main__":
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)

    main(args)

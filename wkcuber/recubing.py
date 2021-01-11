import logging
from typing import List, Tuple

import wkw
import numpy as np
from argparse import ArgumentParser
from itertools import product

from .metadata import detect_bbox

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ensure_wkw,
    add_distribution_flags,
    setup_logging,
    get_executor_for_args,
    wait_and_ensure_success,
)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "source_path", help="Directory containing the datasource properties."
    )

    parser.add_argument(
        "target_path", help="Output directory for the generated dataset."
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

    parser.add_argument(
        "--wkw_file_len", help="Target file length (default 32)", type=int, default=32
    )

    parser.add_argument(
        "--no_compression",
        help="Use compression, default false",
        type=bool,
        default=False,
    )

    add_verbose_flag(parser)
    add_distribution_flags(parser)

    return parser


def next_lower_divisible_by(number: int, divisor: int) -> int:
    remainder = number % divisor
    return number - remainder


def next_higher_divisible_by(number: int, divisor: int) -> int:
    remainder = number % divisor
    return number - remainder + divisor


def recube(
    source_path: str,
    target_path: str,
    layer_name: str,
    dtype: str,
    wkw_file_len: int = 32,
    compression: bool = True,
) -> None:
    if compression:
        block_type = wkw.Header.BLOCK_TYPE_LZ4
    else:
        block_type = wkw.Header.BLOCK_TYPE_RAW

    target_wkw_header = wkw.Header(
        np.dtype(dtype), file_len=wkw_file_len, block_type=block_type
    )
    target_wkw_info = WkwDatasetInfo(target_path, layer_name, 1, target_wkw_header)
    source_wkw_header = wkw.Header(np.dtype(dtype))
    source_wkw_info = WkwDatasetInfo(source_path, layer_name, 1, source_wkw_header)

    ensure_wkw(target_wkw_info)

    bounding_box_dict = detect_bbox(source_wkw_info.dataset_path, layer_name)
    if bounding_box_dict is None:
        raise ValueError("Failed to detect bounding box.")

    bounding_box = (
        bounding_box_dict["topLeft"],
        [
            bounding_box_dict["width"],
            bounding_box_dict["height"],
            bounding_box_dict["depth"],
        ],
    )
    bottom_right = [
        coord + size for coord, size in zip(bounding_box[0], bounding_box[1])
    ]

    wkw_cube_size = wkw_file_len * target_wkw_header.block_len

    outer_bounding_box_tl = list(
        map(lambda lx: next_lower_divisible_by(lx, wkw_cube_size), bounding_box[0])
    )
    outer_bounding_box_br = list(
        map(lambda lx: next_higher_divisible_by(lx, wkw_cube_size), bottom_right)
    )
    outer_bounding_box_size = [
        outer_bounding_box_br[0] - outer_bounding_box_tl[0],
        outer_bounding_box_br[1] - outer_bounding_box_tl[1],
        outer_bounding_box_br[2] - outer_bounding_box_tl[2],
    ]

    target_cube_addresses = product(
        range(0, outer_bounding_box_size[0], wkw_cube_size),
        range(0, outer_bounding_box_size[1], wkw_cube_size),
        range(0, outer_bounding_box_size[2], wkw_cube_size),
    )

    with get_executor_for_args(args) as executor:
        job_args = []
        for target_cube_xyz in target_cube_addresses:
            job_args.append(
                (
                    source_wkw_info,
                    target_wkw_info,
                    outer_bounding_box_size,
                    outer_bounding_box_tl,
                    wkw_cube_size,
                    target_cube_xyz,
                )
            )
        wait_and_ensure_success(executor.map_to_futures(recubing_cube_job, job_args))

    logging.info(f"{layer_name} successfully resampled!")


def recubing_cube_job(
    args: Tuple[
        WkwDatasetInfo, WkwDatasetInfo, List[int], List[int], int, Tuple[int, int, int]
    ]
) -> None:
    (
        source_wkw_info,
        target_wkw_info,
        _outer_bounding_box_size,
        outer_bounding_box_tl,
        wkw_cube_size,
        target_cube_xyz,
    ) = args

    with open_wkw(source_wkw_info) as source_wkw_dataset:
        with open_wkw(target_wkw_info) as target_wkw_dataset:
            top_left = [
                outer_bounding_box_tl[0] + target_cube_xyz[0],
                outer_bounding_box_tl[1] + target_cube_xyz[1],
                outer_bounding_box_tl[2] + target_cube_xyz[2],
            ]

            logging.info("Writing at {}".format(top_left))

            data_cube = source_wkw_dataset.read(
                top_left, (wkw_cube_size, wkw_cube_size, wkw_cube_size)
            )

            target_wkw_dataset.write(top_left, data_cube)


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)

    recube(
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        args.wkw_file_len,
        not args.no_compression,
    )

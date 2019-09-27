import time
import logging
import wkw
import numpy as np
from argparse import ArgumentParser
from os import path
import nibabel as nib

from .utils import (
    add_verbose_flag,
    open_wkw,
    WkwDatasetInfo,
    ensure_wkw,
    add_distribution_flags,
    setup_logging,
    infer_bounding_box,
    BLOCK_LEN
)


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("source_path", help="Directory containing the datasource properties.")

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
        "--wkw_file_len",
        help="Target file length (default 32)",
        type=int,
        default=32,
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


def next_lower_divisible_by(number, divisor) -> int:
    remainder = number % divisor
    return number - remainder


def next_higher_divisible_by(number, divisor) -> int:
    remainder = number % divisor
    return number - remainder + divisor


def recube(source_path, target_path, layer_name, dtype, wkw_file_len=32, compression=True):
    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    source_wkw_info = WkwDatasetInfo(source_path, layer_name, dtype, 1)

    if compression:
        block_type = wkw.Header.BLOCK_TYPE_LZ4
    else:
        block_type = wkw.Header.BLOCK_TYPE_RAW

    ensure_wkw(target_wkw_info, file_len=wkw_file_len, block_type=block_type)

    bounding_box = infer_bounding_box(source_wkw_info)
    bottom_right = [coord + size for coord, size in zip(bounding_box[0], bounding_box[1])]

    outer_bounding_box_tl = list(map(lambda lx: next_lower_divisible_by(lx, wkw_file_len * BLOCK_LEN), bounding_box[0]))
    outer_bounding_box_br = list(map(lambda lx: next_higher_divisible_by(lx, wkw_file_len * BLOCK_LEN), bottom_right))
    outer_bounding_box_size = [outer_bounding_box_br[0] - outer_bounding_box_tl[0],
                               outer_bounding_box_br[1] - outer_bounding_box_tl[1],
                               outer_bounding_box_br[2] - outer_bounding_box_tl[2]]

    with open_wkw(source_wkw_info) as source_wkw_dataset:
        for x in range(0, outer_bounding_box_size[0], wkw_file_len * BLOCK_LEN):
            for y in range(0, outer_bounding_box_size[1], wkw_file_len * BLOCK_LEN):
                for z in range(0, outer_bounding_box_size[2], wkw_file_len * BLOCK_LEN):
                    logging.info("Reading at offset {}, {}, {}".format(x, y, z))

                    top_left = [outer_bounding_box_tl[0] + x,
                                outer_bounding_box_tl[1] + y,
                                outer_bounding_box_tl[2] + z]
                    logging.info("Writing at {}".format(top_left))
                    data_cube = source_wkw_dataset.read(
                        top_left, (wkw_file_len * BLOCK_LEN, wkw_file_len * BLOCK_LEN, wkw_file_len * BLOCK_LEN)
                    )
                    target_wkw_dataset = open_wkw(target_wkw_info, file_len=wkw_file_len, block_type=block_type)
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
        not args.no_compression
    )
import time
import logging
import numpy as np
from typing import Dict, Tuple, List
import os
from glob import glob
import re
from math import floor, ceil, log10
from argparse import ArgumentTypeError

from wkcuber.utils import (
    get_chunks,
    ensure_wkw,
    open_wkw,
    WkwDatasetInfo,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
    get_regular_chunks,
)
from wkcuber.cubing import create_parser as create_cubing_parser, read_image_file
from wkcuber.image_readers import image_reader

BLOCK_LEN = 32
PADDING_FILE_NAME = "/"


# similar to ImageJ https://imagej.net/BigStitcher_StackLoader#File_pattern
def check_input_pattern(input_pattern: str) -> str:
    x_match = re.search("{x+}", input_pattern)
    y_match = re.search("{y+}", input_pattern)
    z_match = re.search("{z+}", input_pattern)

    if x_match is None or y_match is None or z_match is None:
        raise ArgumentTypeError("{} is not a valid pattern".format(input_pattern))

    return input_pattern


def replace_coordinates(
    pattern: str, coord_ids_with_replacement_info: Dict[str, Tuple[int, int]]
) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        coord = occurrence[1]
        if coord in coord_ids_with_replacement_info:
            number_of_digits = (
                len(occurrence) - 2 - coord_ids_with_replacement_info[coord][1]
            )
            if number_of_digits > 1:
                format_str = "0" + str(number_of_digits) + "d"
            else:
                format_str = "d"
            pattern = pattern.replace(
                occurrence,
                format(coord_ids_with_replacement_info[coord][0], format_str),
                1,
            )
    return pattern


def replace_coordinates_with_glob_regex(pattern: str, coord_ids: Dict[str, int]) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        coord = occurrence[1]
        if coord in coord_ids:
            number_of_digits = len(occurrence) - 2 - coord_ids[coord]
            pattern = pattern.replace(occurrence, "[0-9]" * number_of_digits, 1)
    return pattern


def get_digit_numbers_for_dimension(pattern):
    x_number = 0
    y_number = 0
    z_number = 0
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        if occurrence[1] == "x":
            x_number = max(x_number, len(occurrence) - 2)
        if occurrence[1] == "y":
            y_number = max(y_number, len(occurrence) - 2)
        if occurrence[1] == "z":
            z_number = max(z_number, len(occurrence) - 2)
    return x_number, y_number, z_number


def detect_interval_for_dimensions(
    file_path_pattern: str,
    x_decimal_length: int,
    y_decimal_length: int,
    z_decimal_length: int,
) -> int:
    x_min = None
    y_min = None
    z_min = None
    x_max = 0
    y_max = 0
    z_max = 0
    arbitrary_file = None
    file_count = 0
    for x in range(x_decimal_length):
        for y in range(y_decimal_length):
            for z in range(z_decimal_length):
                specific_pattern = replace_coordinates_with_glob_regex(
                    file_path_pattern, {"z": z, "y": y, "x": x}
                )
                found_files = glob(specific_pattern)
                file_count += len(found_files)
                for file in found_files:
                    arbitrary_file = file
                    occurrences = re.findall("({x+}|{y+}|{z+})", file_path_pattern)
                    index_offset_caused_by_brackets_and_specific_length = 0
                    for occurrence in occurrences:
                        # update the offset since the pattern and the file path have different length
                        occurrence_begin_index = (
                            file_path_pattern.index(occurrence)
                            - index_offset_caused_by_brackets_and_specific_length
                        )
                        index_offset_caused_by_brackets_and_specific_length += 2
                        if occurrence[1] == "x":
                            occurrence_end_index = (
                                occurrence_begin_index + x_decimal_length - x
                            )
                            index_offset_caused_by_brackets_and_specific_length += x
                            coordinate_value = int(
                                file[occurrence_begin_index:occurrence_end_index]
                            )
                            x_min = (
                                x_min
                                if x_min and x_min < coordinate_value
                                else coordinate_value
                            )
                            x_max = (
                                x_max if x_max > coordinate_value else coordinate_value
                            )
                        elif occurrence[1] == "y":
                            occurrence_end_index = (
                                occurrence_begin_index + y_decimal_length - y
                            )
                            index_offset_caused_by_brackets_and_specific_length += y
                            coordinate_value = int(
                                file[occurrence_begin_index:occurrence_end_index]
                            )
                            y_min = (
                                y_min
                                if y_min and y_min < coordinate_value
                                else coordinate_value
                            )
                            y_max = (
                                y_max if y_max > coordinate_value else coordinate_value
                            )
                        else:
                            occurrence_end_index = (
                                occurrence_begin_index + z_decimal_length - z
                            )
                            index_offset_caused_by_brackets_and_specific_length += z
                            coordinate_value = int(
                                file[occurrence_begin_index:occurrence_end_index]
                            )
                            z_min = (
                                z_min
                                if z_min and z_min < coordinate_value
                                else coordinate_value
                            )
                            z_max = (
                                z_max if z_max > coordinate_value else coordinate_value
                            )

    return z_min, z_max, y_min, y_max, x_min, x_max, arbitrary_file, file_count


def find_file_with_dimensions(
    file_path_pattern: str,
    x_value: int,
    y_value: int,
    z_value: int,
    x_decimal_length: int,
    y_decimal_length: int,
    z_decimal_length: int,
) -> str:
    found_path = None
    # optimize the bounds
    upper_bound_x = min(floor(log10(x_decimal_length)) + 1, x_decimal_length)
    upper_bound_y = min(floor(log10(y_decimal_length)) + 1, y_decimal_length)
    upper_bound_z = min(floor(log10(z_decimal_length)) + 1, z_decimal_length)

    # try to find the file with all combinations of number lengths
    for z_missing_number_length in range(upper_bound_z):
        for y_missing_number_length in range(upper_bound_y):
            for x_missing_number_length in range(upper_bound_x):
                file_path = replace_coordinates(
                    file_path_pattern,
                    {
                        "z": (z_value, z_missing_number_length),
                        "y": (y_value, y_missing_number_length),
                        "x": (x_value, x_missing_number_length),
                    },
                )
                if os.path.isfile(file_path):
                    # set the file as found and break out of the
                    return file_path

    return None


def tile_cubing_job(
    target_wkw_info,
    z_batches,
    input_path_pattern,
    batch_size,
    tile_size,
    y_min,
    y_max,
    x_min,
    x_max,
    x_decimal_length,
    y_decimal_length,
    z_decimal_length,
):
    if len(z_batches) == 0:
        return

    with open_wkw(target_wkw_info) as target_wkw:
        # Iterate over the z batches
        # Batching is useful to utilize IO more efficiently
        for z_batch in get_chunks(z_batches, batch_size):
            try:
                ref_time = time.time()
                logging.info("Cubing z={}-{}".format(z_batch[0], z_batch[-1]))

                for x in range(x_min, x_max + 1):
                    for y in range(y_min, y_max + 1):
                        ref_time2 = time.time()
                        buffer = []
                        for z in z_batch:
                            # Read file if exists or zeros instead
                            file = find_file_with_dimensions(
                                input_path_pattern,
                                x,
                                y,
                                z,
                                x_decimal_length,
                                y_decimal_length,
                                z_decimal_length,
                            )
                            if file:
                                image = read_image_file(file, target_wkw_info.dtype)
                                image = np.squeeze(image)
                                buffer.append(image)
                            else:
                                # add zeros instead
                                buffer.append(
                                    np.squeeze(
                                        np.zeros(tile_size, dtype=target_wkw_info.dtype)
                                    )
                                )

                        buffer = np.stack(buffer, axis=2)
                        # transpose if the data have a color channel
                        if len(buffer.shape) == 4:
                            buffer = np.transpose(buffer, (3, 0, 1, 2))
                        # Write buffer to target if not empty
                        if np.any(buffer != 0):
                            target_wkw.write(
                                [x * tile_size[0], y * tile_size[1], z_batch[0]], buffer
                            )
                        logging.debug(
                            "Cubing of z={}-{} x={} y={} took {:.8f}s".format(
                                z_batch[0], z_batch[-1], x, y, time.time() - ref_time2
                            )
                        )
                logging.debug(
                    "Cubing of z={}-{} took {:.8f}s".format(
                        z_batch[0], z_batch[-1], time.time() - ref_time
                    )
                )
            except Exception as exc:
                logging.error(
                    "Cubing of z={}-{} failed with {}".format(
                        z_batch[0], z_batch[-1], exc
                    )
                )
                raise exc


def tile_cubing(
    target_path, layer_name, dtype, batch_size, input_path_pattern, args=None
):
    x_decimal_length, y_decimal_length, z_decimal_length = get_digit_numbers_for_dimension(
        input_path_pattern
    )
    z_min, z_max, y_min, y_max, x_min, x_max, arbitraty_file, file_count = detect_interval_for_dimensions(
        input_path_pattern, x_decimal_length, y_decimal_length, z_decimal_length
    )

    if not arbitraty_file:
        logging.error("No source files found")
        return

    # Determine tile size from first matching file
    tile_size = image_reader.read_dimensions(arbitraty_file)
    num_channels = image_reader.read_channel_count(arbitraty_file)
    tile_size = (tile_size[0], tile_size[1], num_channels)
    logging.info(
        "Found source files: count={} with tile_size={}x{}".format(
            file_count, tile_size[0], tile_size[1]
        )
    )

    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    ensure_wkw(target_wkw_info, num_channels=num_channels)
    with get_executor_for_args(args) as executor:
        futures = []
        # Iterate over all z batches
        for z_batch in get_regular_chunks(z_min, z_max, BLOCK_LEN):
            futures.append(
                executor.submit(
                    tile_cubing_job,
                    target_wkw_info,
                    list(z_batch),
                    input_path_pattern,
                    batch_size,
                    tile_size,
                    y_min,
                    y_max,
                    x_min,
                    x_max,
                    x_decimal_length,
                    y_decimal_length,
                    z_decimal_length,
                )
            )
        wait_and_ensure_success(futures)


def create_parser():
    parser = create_cubing_parser()

    parser.add_argument(
        "--input_path_pattern",
        help="Path to input images e.g. path_{xxxxx}_{yyyyy}_{zzzzz}/image.tiff. "
        "The number of signs indicate the longest number in the dimension to the base of 10.",
        type=check_input_pattern,
        default="{zzzzzzzzzzzzzzz}/{yyyyyyyyyyyyyyy}/{xxxxxxxxxxxxxxx}.jpg",
    )
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    input_path_pattern = os.path.join(args.source_path, args.input_path_pattern)

    tile_cubing(
        args.target_path,
        args.layer_name,
        args.dtype,
        int(args.batch_size),
        input_path_pattern,
        args,
    )

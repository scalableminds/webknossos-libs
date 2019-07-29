import time
import logging
import numpy as np
from typing import List, Dict
import os
from glob import glob
import re
import wkw
from argparse import ArgumentParser, ArgumentTypeError
from os import path, listdir
from PIL import Image

from wkcuber.utils import (
    get_chunks,
    get_regular_chunks,
    find_files,
    ensure_wkw,
    open_wkw,
    WkwDatasetInfo,
    add_distribution_flags,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from wkcuber.cubing import create_parser as create_cubing_parser, read_image_file
from wkcuber.image_readers import image_reader

BLOCK_LEN = 32
CUBE_REGEX = re.compile("(\d+)/(\d+)\.([a-zA-Z]{3,4})$")


# similar to ImageJ https://imagej.net/BigStitcher_StackLoader#File_pattern
def check_input_pattern(input_pattern: str) -> str :
    x_match = re.search("{x+}", input_pattern)
    y_match = re.search("{y+}", input_pattern)
    z_match = re.search("{z+}", input_pattern)

    if x_match is None or y_match is None or z_match is None:
        raise ArgumentTypeError("{} is not a valid pattern".format(input_pattern))

    return input_pattern


def replace_coordinate(pattern: str, coord_id: str, coord: int) -> str:
    occurrences = re.findall("{" + coord_id + "+}", pattern)
    for occurrence in occurrences:
        number_of_digits = len(occurrence) - 2
        if number_of_digits > 1:
            format_str = "0"+ str(number_of_digits) + "d"
        else:
            format_str = "d"
        pattern = pattern.replace(occurrence, format(coord, format_str), 1)
    return pattern


def replace_coordinates_with_regex(pattern: str, coord_ids: Dict[str, int]) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        if occurrence[1] in coord_ids:
            number_of_digits = len(occurrence) - 2 - coord_ids[occurrence[1]]
            pattern = pattern.replace(occurrence, "[0-9]" * number_of_digits, 1)
    return pattern

def get_digit_numbers_for_dimension(pattern):
    x_number = 0
    y_number = 0
    z_number = 0
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        if occurrence[1] == 'x':
            x_number = max(x_number, len(occurrence) - 2)
        if occurrence[1] == 'y':
            y_number = max(y_number, len(occurrence) - 2)
        if occurrence[1] == 'z':
            z_number = max(z_number, len(occurrence) - 2)
    return x_number, y_number, z_number


def detect_interval_for_dimensions(file_path_pattern: str, x_decimal_length: int, y_decimal_length: int, z_decimal_length: int) -> List[str]:
    x_min = None
    y_min = None
    z_min = None
    x_max = 0
    y_max = 0
    z_max = 0
    all_files_unordered = []
    for x in range(x_decimal_length):
        for y in range(y_decimal_length):
            for z in range(z_decimal_length):
                specific_pattern = replace_coordinates_with_regex(file_path_pattern, {'z': z, 'y': y, 'x': x})
                found_files = glob(specific_pattern)
                all_files_unordered.extend(found_files)
                for file in found_files:
                    occurrences = re.findall("({x+}|{y+}|{z+})", file_path_pattern)
                    index_offset_caused_by_brackets_and_specific_length = 0
                    for occurrence in occurrences:
                        # update the offset since the pattern and the file path have different length
                        occurrence_begin_index = file_path_pattern.index(occurrence) - index_offset_caused_by_brackets_and_specific_length
                        index_offset_caused_by_brackets_and_specific_length = index_offset_caused_by_brackets_and_specific_length + 2
                        if occurrence[1] == 'x':
                            occurrence_end_index = occurrence_begin_index + x_decimal_length - x
                            index_offset_caused_by_brackets_and_specific_length = index_offset_caused_by_brackets_and_specific_length + x
                            coordinate_value = int(file[occurrence_begin_index:occurrence_end_index])
                            x_min = x_min if x_min and x_min < coordinate_value else coordinate_value
                            x_max = x_max if x_max > coordinate_value else coordinate_value
                        elif occurrence[1] == 'y':
                            occurrence_end_index = occurrence_begin_index + y_decimal_length - y
                            index_offset_caused_by_brackets_and_specific_length = index_offset_caused_by_brackets_and_specific_length + y
                            coordinate_value = int(file[occurrence_begin_index:occurrence_end_index])
                            y_min = y_min if y_min and y_min < coordinate_value else coordinate_value
                            y_max = y_max if y_max > coordinate_value else coordinate_value
                        else:
                            occurrence_end_index = occurrence_begin_index + z_decimal_length - z
                            index_offset_caused_by_brackets_and_specific_length = index_offset_caused_by_brackets_and_specific_length + z
                            coordinate_value = int(file[occurrence_begin_index:occurrence_end_index])
                            z_min = z_min if z_min and z_min < coordinate_value else coordinate_value
                            z_max = z_max if z_max > coordinate_value else coordinate_value

    return z_min, z_max, y_min, y_max, x_min, x_max

def find_source_filenames_by_pattern(file_path_pattern: str) -> List[str]:
    x_decimal_length, y_decimal_length, z_decimal_length = get_digit_numbers_for_dimension(file_path_pattern)
    z_min, z_max, y_min, y_max, x_min, x_max = detect_interval_for_dimensions(file_path_pattern, x_decimal_length, y_decimal_length, z_decimal_length)

    ordered_files = []
    for z in range(z_min, z_max + 1):
        z_pattern = replace_coordinate(file_path_pattern, 'z', z)
        files_in_z_dimension = []
        for y in range(y_min, y_max + 1):
            z_y_pattern = replace_coordinate(z_pattern, 'y', y)
            files_in_z_y_dimension = []
            for x in range(x_min, x_max + 1):
                z_y_x_pattern = replace_coordinate(z_y_pattern, 'x', x)
                files_in_z_y_dimension.append(z_y_x_pattern)
            files_in_z_dimension.append(files_in_z_y_dimension)
        ordered_files.append(files_in_z_dimension)

    return z_min, y_min, x_min, ordered_files


def tile_cubing_job(target_wkw_info, batch_start_index, batch_ordered_files, batch_size, tile_size,
                    z_offset, y_offset, x_offset):
    if len(batch_ordered_files) == 0:
        return

    with open_wkw(target_wkw_info) as target_wkw:
        # Iterate over the z batches
        # Batching is useful to utilize IO more efficiently
        for z_batch in get_chunks(batch_ordered_files, batch_size):
            try:
                ref_time = time.time()
                logging.info("Cubing z={}-{}".format(z_batch[0], z_batch[-1]))

                for x in range(len(z_batch[0][0])):
                    for y in range(len(z_batch[0])):
                        ref_time2 = time.time()
                        buffer = []
                        for z in range(len(z_batch)):
                            # Read file if exists or zeros instead
                            if os.path.isfile(z_batch[z][y][x]):
                                image = read_image_file(z_batch[z][y][x], target_wkw_info.dtype)
                                image = np.squeeze(image)
                                buffer.append(image)
                            else:
                                logging.warning(f"File: {z_batch[z][y][x]} expected but not found. The file will be skipped. "
                                                f"This might produce unexpected results.")
                                buffer.append(
                                    np.zeros(tile_size, dtype=target_wkw_info.dtype)
                                )
                        # Write buffer to target
                        if np.any(buffer != 0):
                            buffer = np.stack(buffer)
                            # transpose if the data have a color channel
                            if(len(buffer.shape) == 4):
                                buffer = np.transpose(buffer, (3, 0, 1, 2))
                            target_wkw.write(
                                [x * tile_size[0] + x_offset, y * tile_size[1] + y_offset, z_offset + batch_start_index], buffer
                            )
                        logging.debug(
                            "Cubing of z={}-{} x={} y={} took {:.8f}s".format(
                                z_offset, z_offset + len(z_batch), x + x_offset, y + y_offset, time.time() - ref_time2
                            )
                        )
                logging.debug(
                    "Cubing of z={}-{} took {:.8f}s".format(
                        z_offset, z_offset + len(z_batch), time.time() - ref_time
                    )
                )
            except Exception as exc:
                logging.error(
                    "Cubing of z={}-{} failed with {}".format(
                        z_offset, z_offset + len(z_batch), exc
                    )
                )
                raise exc


def tile_cubing(
    target_path, layer_name, dtype, batch_size, input_path_pattern, args=None
):

    z_min, x_min, y_min, ordered_files = find_source_filenames_by_pattern(input_path_pattern)
    if len(ordered_files) == 0:
        logging.error("No source files found")
        return

    # Determine tile size from first matching file
    tile_size = image_reader.read_dimensions(
        ordered_files[0][0][0]
    )
    num_channels = image_reader.read_channel_count(ordered_files[0][0][0])
    logging.info(
        "Found source files: count={} tile_size={}x{}".format(
            sum(len(z_files) for z_files in ordered_files), tile_size[0], tile_size[1]
        )
    )

    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    ensure_wkw(target_wkw_info, num_channels=num_channels)
    with get_executor_for_args(args) as executor:
        futures = []
        # Iterate over all z batches
        for z_start_index in range(0, len(ordered_files), BLOCK_LEN):
            futures.append(
                executor.submit(
                    tile_cubing_job,
                    target_wkw_info,
                    z_start_index,
                    ordered_files[z_start_index: z_start_index + batch_size],
                    batch_size,
                    tile_size,
                    z_min + z_start_index,
                    y_min,
                    z_min
                )
            )
        wait_and_ensure_success(futures)

def create_parser():
    parser = create_cubing_parser()

    parser.add_argument("--input_path_pattern",
                        help="Path to input images e.g. path_{x}_{y}_{z}/image.tiff",
                        type=check_input_pattern,
                        default="{z}/{y}/{x}.tiff")
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

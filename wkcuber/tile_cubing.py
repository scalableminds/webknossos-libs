import time
import logging
import numpy as np
from typing import Dict, Tuple
import os
from glob import glob
import re
from argparse import ArgumentTypeError

from wkcuber.utils import (
    get_chunks,
    ensure_wkw,
    open_wkw,
    WkwDatasetInfo,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
)
from wkcuber.cubing import create_parser as create_cubing_parser, read_image_file
from wkcuber.image_readers import image_reader

BLOCK_LEN = 32
CUBE_REGEX = re.compile("(\d+)/(\d+)\.([a-zA-Z]{3,4})$")


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
    for x in range(x_decimal_length):
        for y in range(y_decimal_length):
            for z in range(z_decimal_length):
                specific_pattern = replace_coordinates_with_glob_regex(
                    file_path_pattern, {"z": z, "y": y, "x": x}
                )
                found_files = glob(specific_pattern)
                for file in found_files:
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

    return z_min, z_max, y_min, y_max, x_min, x_max


def list_all_source_files_ordered(
    file_path_pattern: str,
    z_min: int,
    z_max: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    x_decimal_length: int,
    y_decimal_length: int,
    z_decimal_length: int,
):
    ordered_files = []
    number_of_files_found = 0
    # scan for files in the whole range
    for z in range(z_min, z_max + 1):
        files_in_z_dimension = []
        for y in range(y_min, y_max + 1):
            files_in_z_y_dimension = []
            for x in range(x_min, x_max + 1):
                found_path = None

                # try to find the file with all combinations of number lengths
                for z_missing_number_length in range(x_decimal_length):
                    for y_missing_number_length in range(y_decimal_length):
                        for x_missing_number_length in range(z_decimal_length):
                            file_path = replace_coordinates(
                                file_path_pattern,
                                {
                                    "z": (z, z_missing_number_length),
                                    "y": (y, y_missing_number_length),
                                    "x": (x, x_missing_number_length),
                                },
                            )

                            if os.path.isfile(file_path):
                                # set the file as found and break out of the
                                number_of_files_found += 1
                                found_path = file_path
                                break
                        if found_path:
                            break
                    if found_path:
                        break
                if not found_path:
                    # still get a default one since the script needs a file to maintain the order
                    found_path = file_path
                # create a list for each sub-dimension
                files_in_z_y_dimension.append(found_path)
            files_in_z_dimension.append(files_in_z_y_dimension)
        ordered_files.append(files_in_z_dimension)

    return ordered_files, number_of_files_found


def tile_cubing_job(
    target_wkw_info,
    batch_ordered_files,
    batch_size,
    tile_size,
    z_offset,
    y_offset,
    x_offset,
):
    if len(batch_ordered_files) == 0:
        return

    with open_wkw(target_wkw_info) as target_wkw:
        # Iterate over the z batches
        # Batching is useful to utilize IO more efficiently
        z_batch_offset = 0
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
                                image = read_image_file(
                                    z_batch[z][y][x], target_wkw_info.dtype
                                )
                                image = np.squeeze(image)
                                buffer.append(image)
                            else:
                                logging.warning(
                                    f"File: {z_batch[z][y][x]} expected but not found. The file will be skipped. "
                                    f"This might produce unexpected results."
                                )
                                buffer.append(
                                    np.squeeze(
                                        np.zeros(tile_size, dtype=target_wkw_info.dtype)
                                    )
                                )
                        # Write buffer to target
                        if np.any(buffer != 0):
                            buffer = np.stack(buffer)
                            # transpose if the data have a color channel
                            if len(buffer.shape) == 4:
                                buffer = np.transpose(buffer, (3, 0, 1, 2))
                            target_wkw.write(
                                [
                                    x * tile_size[0] + x_offset,
                                    y * tile_size[1] + y_offset,
                                    z_offset + z_batch_offset,
                                ],
                                buffer,
                            )
                        logging.debug(
                            "Cubing of z={}-{} x={} y={} took {:.8f}s".format(
                                z_offset + z_batch_offset,
                                z_offset + z_batch_offset + len(z_batch),
                                x + x_offset,
                                y + y_offset,
                                time.time() - ref_time2,
                            )
                        )
                logging.debug(
                    "Cubing of z={}-{} took {:.8f}s".format(
                        z_offset + z_batch_offset,
                        z_offset + z_batch_offset + len(z_batch),
                        time.time() - ref_time,
                    )
                )
            except Exception as exc:
                logging.error(
                    "Cubing of z={}-{} failed with {}".format(
                        z_offset + z_batch_offset,
                        z_offset + z_batch_offset + len(z_batch),
                        exc,
                    )
                )
                raise exc
            z_batch_offset += len(z_batch)


def tile_cubing(
    target_path, layer_name, dtype, batch_size, input_path_pattern, args=None
):
    logging.info("Searching for files with the input pattern. This might take a while.")
    x_decimal_length, y_decimal_length, z_decimal_length = get_digit_numbers_for_dimension(
        input_path_pattern
    )
    z_min, z_max, y_min, y_max, x_min, x_max = detect_interval_for_dimensions(
        input_path_pattern, x_decimal_length, y_decimal_length, z_decimal_length
    )

    ordered_files, number_of_files_found = list_all_source_files_ordered(
        input_path_pattern,
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
        x_decimal_length,
        y_decimal_length,
        z_decimal_length,
    )
    if len(ordered_files) == 0:
        logging.error("No source files found")
        return

    # Determine tile size from first matching file
    tile_size = image_reader.read_dimensions(ordered_files[0][0][0])
    num_channels = image_reader.read_channel_count(ordered_files[0][0][0])
    tile_size = (tile_size[0], tile_size[1], num_channels)
    logging.info(
        "Found source files: count={} with tile_size={}x{} and expected to find for contiguous data count={}".format(
            number_of_files_found,
            tile_size[0],
            tile_size[1],
            len(ordered_files) * len(ordered_files[0]) * len(ordered_files[0][0]),
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
                    ordered_files[z_start_index : z_start_index + BLOCK_LEN],
                    batch_size,
                    tile_size,
                    z_min + z_start_index,
                    y_min,
                    z_min,
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

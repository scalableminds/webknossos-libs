import time
import logging
import numpy as np
from typing import List
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


def replace_coordinates_with_regex(pattern: str, coord_ids: List[str]) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrences in occurrences:
        if occurrences[1] in coord_ids:
            number_of_digits = len(occurrences) - 2
            pattern = pattern.replace(occurrences, "[0-9]" * number_of_digits, 1)
    return pattern


def find_source_filenames_by_pattern(file_path_pattern: str) -> List[str]:
    search_regex = replace_coordinates_with_regex(file_path_pattern, ['z', 'y', 'x'])
    all_files_unordered = glob(search_regex)
    z_min, z_max, y_min, y_max, x_min, x_max = detect_interval_for_dimensions(file_path_pattern, all_files_unordered)

    ordered_files = []
    for z in range(z_min, z_max + 1):
        z_pattern = replace_coordinate(file_path_pattern, 'z', z)
        files_in_this_z_dimension = []
        for y in range(y_min, y_max + 1):
            z_y_pattern = replace_coordinate(z_pattern, 'y', y)
            for x in range(x_min, x_max + 1):
                z_y_x_pattern = replace_coordinate(z_y_pattern, 'x', x)
        ordered_files.append(files_in_this_z_dimension)

    return z_min, z_max, ordered_files

def detect_interval_for_dimensions(file_path_pattern: str, files):
    x_min = None
    y_min = None
    z_min = None
    x_max = 0
    y_max = 0
    z_max = 0
    for file in files:
        occurrences = re.findall("({x+}|{y+}|{z+})", file_path_pattern)
        index_offset_caused_by_brackets = 0
        for occurrence in occurrences:
            occurrence_begin_index = file_path_pattern.index(occurrence) - index_offset_caused_by_brackets
            occurrence_end_index = occurrence_begin_index + len(occurrence) - 2
            index_offset_caused_by_brackets = index_offset_caused_by_brackets + 2
            coordinate_value = int(file[occurrence_begin_index:occurrence_end_index])
            if occurrence[1] == 'x':
                x_min = x_min if x_min and x_min < coordinate_value else coordinate_value
                x_max = x_max if x_max > coordinate_value else coordinate_value
            elif occurrence[1] == 'y':
                y_min = y_min if y_min and y_min < coordinate_value else coordinate_value
                y_max = y_max if y_max > coordinate_value else coordinate_value
            else:
                z_min = z_min if z_min and z_min < coordinate_value else coordinate_value
                z_max = z_max if z_max > coordinate_value else coordinate_value

    return z_min, z_max, y_min, y_max, x_min, x_max


def determine_dimensions_order_in_pattern(file_path_pattern: str) -> List[str]:
    dimension_order = []
    occurrences = re.findall("({x+}|{y+}|{z+})", file_path_pattern)
    for occurrence in occurrences:
        dimension_order.append(occurrence[1])
    return dimension_order

def find_source_sections(source_path):
    section_folders = [
        f for f in listdir(source_path) if path.isdir(path.join(source_path, f))
    ]
    section_folders = [path.join(source_path, s) for s in section_folders]
    section_folders.sort()
    return section_folders


def parse_tile_file_name(filename):
    m = CUBE_REGEX.search(filename)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), m.group(3))


def find_source_files(source_section_path):
    all_source_files = [
        path.join(source_section_path, s)
        for s in find_files(
            path.join(source_section_path, "**", "*"), image_reader.readers.keys()
        )
    ]

    all_source_files.sort()
    return all_source_files


def tile_cubing_job(target_wkw_info, batch_ordered_files, batch_size, tile_size):
    if len(batch_ordered_files) == 0:
        return

    with open_wkw(target_wkw_info) as target_wkw:
        # Iterate over the z batches
        # Batching is useful to utilize IO more efficiently
        for z_batch in get_chunks(z_batches, batch_size):
            try:
                ref_time = time.time()
                logging.info("Cubing z={}-{}".format(z_batch[0], z_batch[-1]))

                for file in batch_ordered_files:


                # Detect all available tiles in this z batch
                tile_coords = []
                for z in z_batch:
                    tile_coords += [
                        (z,) + parse_tile_file_name(f)
                        for f in find_source_files("{}/{}".format(source_path, z))
                    ]
                # Figure out what x-y combinations are available in this z batch
                xy = sorted(set([(x, y) for z, y, x, _ in tile_coords]))

                # Iterate over all x-y combinations from this z batch
                for x, y in xy:
                    ref_time2 = time.time()
                    buffer = []
                    for z in z_batch:
                        # Find the file extension of the x-y tile in this z
                        ext = next(
                            (
                                ext
                                for _z, _y, _x, ext in tile_coords
                                if _z == z and _y == y and _x == x
                            ),
                            None,
                        )
                        # Read file if exists or zeros instead
                        if ext is not None:
                            image = read_image_file(
                                "{}/{}/{}/{}.{}".format(source_path, z, y, x, ext),
                                target_wkw_info.dtype,
                            )
                            buffer.append(image)
                        else:
                            logging.warning(f"File: {z_y_x_pattern} expected but not found. The file will be skipped. "
                                    f"This might result in unexpected results.")
                            buffer.append(
                                np.zeros(tile_size, dtype=target_wkw_info.dtype)
                            )

                    # Write buffer to target
                    buffer = np.dstack(buffer)
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
    source_path, target_path, layer_name, dtype, batch_size, jobs, input_path_pattern, args=None
):

    z_min, z_max, ordered_files = find_source_filenames_by_pattern(input_path_pattern)
    # Detect available z sections
    #sections = find_source_sections(source_path)
    if len(ordered_files) == 0:
        logging.error("No source files found")
        return
    #min_z = min([int(path.basename(f)) for f in sections])
    #max_z = max([int(path.basename(f)) for f in sections])

    # Determine tile size from first matching file
    tile_size = image_reader.read_dimensions(
        #next(find_files(path.join(source_path, "**", "*"), image_reader.readers.keys()))
        ordered_files[0][0]
    )
    logging.info(
        "Found source files: count={} tile_size={}x{}".format(
            sum(len(z_files) for z_files in ordered_files), tile_size[0], tile_size[1]
        )
    )

    target_wkw_info = WkwDatasetInfo(target_path, layer_name, dtype, 1)
    ensure_wkw(target_wkw_info)
    with get_executor_for_args(args) as executor:
        futures = []
        # Iterate over all z batches
        for z_start_index in range(0, len(ordered_files), batch_size):
            futures.append(
                executor.submit(
                    tile_cubing_job,
                    target_wkw_info,
                    ordered_files[z_start_index: z_start_index + batch_size],
                    batch_size,
                    tile_size,
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
        args.source_path,
        args.target_path,
        args.layer_name,
        args.dtype,
        int(args.batch_size),
        int(args.jobs),
        input_path_pattern,
        args,
    )

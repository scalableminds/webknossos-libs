import logging
from pathlib import Path

import numpy as np
from typing import Dict, Tuple, Union, List, Optional, cast
import os
from glob import glob
import re
from argparse import ArgumentTypeError, ArgumentParser, Namespace

from webknossos.dataset import Dataset, View, COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.geometry import BoundingBox, Vec3Int, Mag

from webknossos.dataset import SegmentationLayer
from webknossos.utils import time_start, time_stop
from .utils import (
    get_chunks,
    get_executor_for_args,
    wait_and_ensure_success,
    setup_logging,
    get_regular_chunks,
)
from .cubing import create_parser as create_cubing_parser
from .cubing import read_image_file
from .image_readers import image_reader

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
    """Replaces the coordinates with a specific length.
    The coord_ids_with_replacement_info is a Dict that maps a dimension
    to a tuple of the coordinate value and the desired length."""
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        coord = occurrence[1]
        if coord in coord_ids_with_replacement_info:
            number_of_digits = coord_ids_with_replacement_info[coord][1]
            format_str = "0" + str(number_of_digits) + "d"
            pattern = pattern.replace(
                occurrence,
                format(coord_ids_with_replacement_info[coord][0], format_str),
                1,
            )
    return pattern


def replace_pattern_to_specific_length_without_brackets(
    pattern: str, coord_ids_with_specific_length: Dict[str, int]
) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        coord = occurrence[1]
        if coord in coord_ids_with_specific_length:
            pattern = pattern.replace(
                occurrence, coord * coord_ids_with_specific_length[coord], 1
            )
    return pattern


def replace_coordinates_with_glob_regex(pattern: str, coord_ids: Dict[str, int]) -> str:
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    for occurrence in occurrences:
        coord = occurrence[1]
        if coord in coord_ids:
            number_of_digits = coord_ids[coord]
            pattern = pattern.replace(occurrence, "[0-9]" * number_of_digits, 1)
    return pattern


def get_digit_counts_for_dimensions(pattern: str) -> Dict[str, int]:
    """Counts how many digits the dimensions x, y and z occupy in the given pattern."""
    occurrences = re.findall("({x+}|{y+}|{z+})", pattern)
    decimal_lengths = {"x": 0, "y": 0, "z": 0}

    for occurrence in occurrences:
        current_dimension = occurrence[1]
        decimal_lengths[current_dimension] = max(
            decimal_lengths[current_dimension], len(occurrence) - 2
        )

    return decimal_lengths


def detect_interval_for_dimensions(
    file_path_pattern: str, decimal_lengths: Dict[str, int]
) -> Tuple[Dict[str, int], Dict[str, int], Optional[Path], int]:
    arbitrary_file = None
    file_count = 0
    # dictionary that maps the dimension string to the current dimension length
    # used to avoid distinction of dimensions with if statements
    current_decimal_length = {"x": 0, "y": 0, "z": 0}
    max_dimensions = {"x": 0, "y": 0, "z": 0}
    min_dimensions: Dict[str, int] = {}

    # find all files by trying all combinations of dimension lengths
    for x in range(decimal_lengths["x"] + 1):
        current_decimal_length["x"] = x
        for y in range(decimal_lengths["y"] + 1):
            current_decimal_length["y"] = y
            for z in range(decimal_lengths["z"] + 1):
                current_decimal_length["z"] = z
                specific_pattern = replace_coordinates_with_glob_regex(
                    file_path_pattern, {"z": z, "y": y, "x": x}
                )
                found_files = glob(specific_pattern)
                file_count += len(found_files)
                for file_name in found_files:
                    if arbitrary_file is None:
                        arbitrary_file = Path(file_name)
                    # Turn a pattern {xxx}/{yyy}/{zzzzzz} for given dimension counts into (e.g., 2, 2, 3) into
                    # something like xx/yy/zzz (note that the curly braces are gone)
                    applied_fpp = replace_pattern_to_specific_length_without_brackets(
                        file_path_pattern, {"x": x, "y": y, "z": z}
                    )

                    # For each dimension, look up where it starts within the applied pattern.
                    # Use that index to look up the actual value within the file name
                    for current_dimension in ["x", "y", "z"]:
                        idx = applied_fpp.index(current_dimension)
                        coordinate_value_str = file_name[
                            idx : idx + current_decimal_length[current_dimension]
                        ]
                        coordinate_value = int(coordinate_value_str)
                        assert coordinate_value
                        min_dimensions[current_dimension] = min(
                            min_dimensions.get(current_dimension, coordinate_value),
                            coordinate_value,
                        )
                        max_dimensions[current_dimension] = max(
                            max_dimensions[current_dimension], coordinate_value
                        )

    return min_dimensions, max_dimensions, arbitrary_file, file_count


def find_file_with_dimensions(
    file_path_pattern: str,
    x_value: int,
    y_value: int,
    z_value: int,
    decimal_lengths: Dict[str, int],
) -> Union[Path, None]:
    file_path_unpadded = Path(
        replace_coordinates(
            file_path_pattern, {"z": (z_value, 0), "y": (y_value, 0), "x": (x_value, 0)}
        )
    )

    file_path_padded = Path(
        replace_coordinates(
            file_path_pattern,
            {
                "z": (z_value, decimal_lengths["z"]),
                "y": (y_value, decimal_lengths["y"]),
                "x": (x_value, decimal_lengths["x"]),
            },
        )
    )

    # the unpadded file pattern has a higher precedence
    if file_path_unpadded.is_file():
        return file_path_unpadded

    if file_path_padded.is_file():
        return file_path_padded

    return None


def tile_cubing_job(
    args: Tuple[
        View,
        List[int],
        str,
        int,
        Tuple[int, int, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        str,
        int,
    ]
) -> int:
    (
        target_view,
        z_batches,
        input_path_pattern,
        batch_size,
        tile_size,
        min_dimensions,
        max_dimensions,
        decimal_lengths,
        dtype,
        num_channels,
    ) = args
    largest_value_in_chunk = 0  # This is used to compute the largest_segmentation_id if it is a segmentation layer

    # Iterate over the z batches
    # Batching is useful to utilize IO more efficiently
    for z_batch in get_chunks(z_batches, batch_size):
        try:
            time_start(f"Cubing of z={z_batch[0]}-{z_batch[-1]}")
            for x in range(min_dimensions["x"], max_dimensions["x"] + 1):
                for y in range(min_dimensions["y"], max_dimensions["y"] + 1):
                    # Allocate a large buffer for all images in this batch
                    # Shape will be (channel_count, x, y, z)
                    # Using fortran order for the buffer, prevents that the data has to be copied in rust
                    buffer_shape = [
                        num_channels,
                        tile_size[0],
                        tile_size[1],
                        len(z_batch),
                    ]
                    buffer = np.empty(buffer_shape, dtype=dtype, order="F")
                    for z in z_batch:
                        # Read file if exists or use zeros instead
                        file_name = find_file_with_dimensions(
                            input_path_pattern, x, y, z, decimal_lengths
                        )
                        if file_name:
                            # read the image
                            image = read_image_file(
                                file_name,
                                target_view.header.voxel_type,
                                z,
                                None,
                                None,
                            )
                        else:
                            # add zeros instead
                            image = np.zeros(
                                tile_size + (1,),
                                dtype=target_view.header.voxel_type,
                            )
                        # The size of a image might be smaller than the buffer, if the tile is at the bottom/right border
                        buffer[
                            :, : image.shape[0], : image.shape[1], z - z_batch[0]
                        ] = image.transpose((2, 0, 1, 3))[:, :, :, 0]

                    if np.any(buffer != 0):
                        offset = (
                            (x - min_dimensions["x"]) * tile_size[0],
                            (y - min_dimensions["y"]) * tile_size[1],
                            z_batch[0] - target_view.global_offset.z,
                        )
                        target_view.write(data=buffer, offset=offset)
                        largest_value_in_chunk = max(
                            largest_value_in_chunk, np.max(buffer)
                        )
            time_stop(f"Cubing of z={z_batch[0]}-{z_batch[-1]}")
        except Exception as exc:
            logging.error(
                "Cubing of z={}-{} failed with: {}".format(z_batch[0], z_batch[-1], exc)
            )
            raise exc
    return largest_value_in_chunk


def tile_cubing(
    target_path: Path,
    layer_name: str,
    batch_size: int,
    input_path_pattern: str,
    scale: Tuple[int, int, int],
    args: Optional[Namespace] = None,
) -> None:
    decimal_lengths = get_digit_counts_for_dimensions(input_path_pattern)
    (
        min_dimensions,
        max_dimensions,
        arbitrary_file,
        file_count,
    ) = detect_interval_for_dimensions(input_path_pattern, decimal_lengths)

    if not arbitrary_file:
        logging.error(
            f"No source files found. Maybe the input_path_pattern was wrong. You provided: {input_path_pattern}"
        )
        return
    # Determine tile size from first matching file
    tile_width, tile_height = image_reader.read_dimensions(arbitrary_file)
    num_z = max_dimensions["z"] - min_dimensions["z"] + 1
    num_x = (max_dimensions["x"] - min_dimensions["x"] + 1) * tile_width
    num_y = (max_dimensions["y"] - min_dimensions["y"] + 1) * tile_height
    x_offset = min_dimensions["x"] * tile_width
    y_offset = min_dimensions["y"] * tile_height
    num_channels = image_reader.read_channel_count(arbitrary_file)
    logging.info(
        "Found source files: count={} with tile_size={}x{}".format(
            file_count, tile_width, tile_height
        )
    )
    if args is None or not hasattr(args, "dtype") or args.dtype is None:
        dtype = image_reader.read_dtype(arbitrary_file)
    else:
        dtype = args.dtype

    target_ds = Dataset(target_path, scale=scale, exist_ok=True)
    is_segmentation_layer = layer_name == "segmentation"
    if is_segmentation_layer:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            SEGMENTATION_CATEGORY,
            dtype_per_channel=dtype,
            num_channels=num_channels,
            largest_segment_id=0,
        )
    else:
        target_layer = target_ds.get_or_add_layer(
            layer_name,
            COLOR_CATEGORY,
            dtype_per_channel=dtype,
            num_channels=num_channels,
        )

    bbox = BoundingBox(
        Vec3Int(x_offset, y_offset, min_dimensions["z"]),
        Vec3Int(num_x, num_y, num_z),
    )
    if target_layer.bounding_box.volume() == 0:
        # If the layer is empty, we want to set the bbox directly because extending it
        # would mean that the bbox would always start at (0, 0, 0)
        target_layer.bounding_box = bbox
    else:
        target_layer.bounding_box = target_layer.bounding_box.extended_by(bbox)

    target_mag_view = target_layer.get_or_add_mag(Mag(1), block_len=BLOCK_LEN)

    with get_executor_for_args(args) as executor:
        job_args = []
        # Iterate over all z batches
        for z_batch in get_regular_chunks(
            min_dimensions["z"], max_dimensions["z"], BLOCK_LEN
        ):
            # The z_batch always starts and ends at a multiple of BLOCK_LEN.
            # However, we only want the part that is inside the bounding box
            z_batch = range(
                max(list(z_batch)[0], target_layer.bounding_box.topleft.z),
                min(list(z_batch)[-1] + 1, target_layer.bounding_box.bottomright.z),
            )
            z_values = list(z_batch)
            job_args.append(
                (
                    target_mag_view.get_view(
                        (x_offset, y_offset, z_values[0]),
                        (num_x, num_y, len(z_values)),
                    ),
                    z_values,
                    input_path_pattern,
                    batch_size,
                    (tile_width, tile_height, num_channels),
                    min_dimensions,
                    max_dimensions,
                    decimal_lengths,
                    dtype,
                    num_channels,
                )
            )

        largest_segment_id_per_chunk = wait_and_ensure_success(
            executor.map_to_futures(tile_cubing_job, job_args),
            f"Tile cubing layer {layer_name}",
        )

        if is_segmentation_layer:
            largest_segment_id = max(largest_segment_id_per_chunk)
            cast(
                SegmentationLayer, target_layer
            ).largest_segment_id = largest_segment_id


def create_parser() -> ArgumentParser:
    parser = create_cubing_parser()

    parser.add_argument(
        "--input_path_pattern",
        help="Path to input images e.g. path_{xxxxx}_{yyyyy}_{zzzzz}/image.tiff. "
        "The number of characters indicate the longest number in the dimension to the base of 10.",
        type=check_input_pattern,
        default="{zzzzzzzzzz}/{yyyyyyyyyy}/{xxxxxxxxxx}.jpg",
    )

    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup_logging(args)
    input_path_pattern = os.path.join(args.source_path, args.input_path_pattern)

    tile_cubing(
        args.target_path,
        args.layer_name,
        int(args.batch_size),
        input_path_pattern,
        args.scale,
        args,
    )

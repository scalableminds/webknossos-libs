import logging
import os
import re
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from cluster_tools import Executor
from webknossos import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    BoundingBox,
    Dataset,
    Mag,
    SegmentationLayer,
    Vec3Int,
    View,
)
from webknossos.dataset.defaults import DEFAULT_CHUNK_SHAPE
from webknossos.utils import time_start, time_stop

from ._internal.image_readers import image_reader
from ._internal.utils import (
    get_chunks,
    get_executor_for_args,
    get_regular_chunks,
    setup_logging,
    setup_warnings,
    wait_and_ensure_success,
)
from .cubing import create_parser as create_cubing_parser
from .cubing import read_image_file

PADDING_FILE_NAME = "/"
COORDINATE_REGEX = re.compile("{(x+|y+|z+)}")
COORDINATES = "x", "y", "z"


# similar to ImageJ https://imagej.net/BigStitcher_StackLoader#File_pattern
def check_input_pattern(input_pattern: str) -> str:
    x_match = re.search("{x+}", input_pattern)
    y_match = re.search("{y+}", input_pattern)
    z_match = re.search("{z+}", input_pattern)

    if x_match is None or y_match is None or z_match is None:
        raise ArgumentTypeError("{} is not a valid pattern".format(input_pattern))

    return input_pattern


def path_from_coordinate_pattern(
    pattern: str, coord_ids_with_replacement_info: Dict[str, Tuple[int, int]]
) -> Path:
    """Formulates a path from the given pattern and coordinate info.

    The coord_ids_with_replacement_info is a Dict that maps a dimension
    to a tuple of the coordinate value and the desired length."""
    path_parts: List[str] = []
    last = 0
    for match in COORDINATE_REGEX.finditer(pattern):
        coord = match.group(1)[0]
        value, number_of_digits = coord_ids_with_replacement_info[coord]
        path_parts.append(pattern[last : match.start()])
        path_parts.append(str(value).zfill(number_of_digits))
        last = match.end()

    path_parts.append(pattern[last:])
    return Path("".join(path_parts))


def parse_coordinate_pattern(pattern: str) -> Tuple[List[re.Pattern], Path]:
    """Creates a list of all subdirectores in pattern as regexes

    Only starts creating regexes from the first path component that contains a
    coordinate template string (eg. {x}), these are returned as a pathlib.Path
    as the second component of the return value.

    Raises ValueError in case the pattern is incorrectly formatted."""
    coord_set = {*COORDINATES}
    regexes: List[re.Pattern] = []
    pattern_parts: List[str] = []
    root_path = Path()
    for dirpattern in Path(pattern).parts:
        last = 0
        for match in COORDINATE_REGEX.finditer(dirpattern):
            coord_string = match.group(1)
            coord = coord_string[0]
            try:
                coord_set.remove(coord)
            except KeyError:
                raise ValueError(
                    f"pattern ({pattern}) contains more than one {coord} coordinate"
                ) from None
            else:
                substring = re.escape(dirpattern[last : match.start()])
                length = str(len(coord_string))
                pattern_parts.extend(
                    [substring, "(?P<", coord, ">[0-9]{1,", length, "})"]
                )
                last = match.end()

        if not pattern_parts:  # no matches found yet
            root_path /= dirpattern
        else:
            pattern_parts.append(re.escape(dirpattern[last:]))
            next_pattern = "".join(pattern_parts)
            regexes.append(re.compile(next_pattern))
            pattern_parts = [next_pattern]
            pattern_parts.append(re.escape(os.sep))

    if coord_set:
        raise ValueError(
            f"pattern ({pattern}) is missing these coordinates: {', '.join(coord_set)}"
        )

    return regexes, root_path


def detect_interval_for_dimensions(
    file_path_pattern: str,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Optional[Path], int]:
    """Searches filesystem for all files that will match the given pattern.

    Returns the needed information to find all matching files, in order:
    - The minimum amount of length of each coordinate, this represents the
        amount of leading zeros that should be added inside shorter filenames
    - The lowest value of indexes of each coordinate.
    - The highest value of indexes of each coordinate.
    - The first matching file found.
    - The total amount of matching files.

    Raises RuntimeError in case padding is not used in a consistent way."""
    arbitrary_file = None
    file_count = 0
    padding_found = {coord: False for coord in COORDINATES}
    min_paddings: Dict[str, int] = {}
    max_dimensions: Dict[str, int] = {}
    min_dimensions: Dict[str, int] = {}

    regexes, root_path = parse_coordinate_pattern(file_path_pattern)
    *dir_regexes, full_rx = regexes
    root_depth = len(root_path.parts)
    target_depth = len(dir_regexes)
    for dirname, dirs, files in os.walk(root_path, followlinks=True):
        dir_path_parts = Path(dirname).parts[root_depth:]
        dir_path = Path(*dir_path_parts)
        depth = len(dir_path_parts)
        if depth != 0:
            try:
                dir_rx = dir_regexes[depth - 1]
            except IndexError:
                dirs.clear()
                continue

            if dir_rx.fullmatch(str(dir_path)) is None:
                dirs.clear()
                continue

        if depth != target_depth:
            continue

        dirs.clear()
        for file in files:
            file_path = dir_path / file
            match = full_rx.fullmatch(str(file_path))
            if match is None:
                continue

            file_count += 1
            if arbitrary_file is None:
                arbitrary_file = root_path / file_path

            for current_dimension in COORDINATES:
                coordinate_value_str = match.group(current_dimension)
                coordinate_value = int(coordinate_value_str)
                length = len(coordinate_value_str)
                is_padded = coordinate_value_str[0] == "0"
                if file_count == 1:
                    min_paddings[current_dimension] = length
                    min_dimensions[current_dimension] = coordinate_value
                    max_dimensions[current_dimension] = coordinate_value
                else:
                    previous_length = min_paddings[current_dimension]
                    if is_padded:
                        if length > previous_length:
                            raise RuntimeError("Inconsistent use of padding found")

                    previous_is_padded = padding_found[current_dimension]
                    if previous_is_padded:
                        if length < previous_length:
                            raise RuntimeError("Inconsistent use of padding found")
                    else:
                        min_paddings[current_dimension] = min(length, previous_length)

                    min_dimensions[current_dimension] = min(
                        coordinate_value, min_dimensions[current_dimension]
                    )
                    max_dimensions[current_dimension] = max(
                        coordinate_value, max_dimensions[current_dimension]
                    )

                if is_padded:
                    padding_found[current_dimension] = True

    return min_paddings, min_dimensions, max_dimensions, arbitrary_file, file_count


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
        min_paddings,
        dtype,
        num_channels,
    ) = args
    largest_value_in_chunk = 0  # This is used to compute the largest_segmentation_id if it is a segmentation layer
    z_offset = target_view.bounding_box.in_mag(target_view.mag).topleft.z

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
                        coordinate_info: Dict[str, Tuple[int, int]] = {}
                        for coord, value in zip(COORDINATES, (x, y, z)):
                            coordinate_info[coord] = value, min_paddings[coord]

                        file_path = path_from_coordinate_pattern(
                            input_path_pattern, coordinate_info
                        )
                        if file_path.exists():
                            # read the image
                            image = read_image_file(
                                file_path,
                                target_view.info.voxel_type,
                                z,
                                None,
                                None,
                            )
                        else:
                            # add zeros instead
                            image = np.zeros(
                                tile_size + (1,),
                                dtype=target_view.info.voxel_type,
                            )
                        # The size of a image might be smaller than the buffer, if the tile is at the bottom/right border
                        buffer[
                            :, : image.shape[0], : image.shape[1], z - z_batch[0]
                        ] = image.transpose((2, 0, 1, 3))[:, :, :, 0]

                    if np.any(buffer != 0):
                        offset = (
                            (x - min_dimensions["x"]) * tile_size[0],
                            (y - min_dimensions["y"]) * tile_size[1],
                            z_batch[0] - z_offset,
                        )
                        target_view.write(data=buffer, relative_offset=offset)
                        largest_value_in_chunk = max(
                            largest_value_in_chunk, np.max(buffer)
                        )
            time_stop(f"Cubing of z={z_batch[0]}-{z_batch[-1]}")
        except Exception as exc:
            raise RuntimeError(
                "Cubing of z={}-{} failed".format(z_batch[0], z_batch[-1])
            ) from exc

    return largest_value_in_chunk


def tile_cubing(
    target_path: Path,
    layer_name: str,
    batch_size: int,
    input_path_pattern: str,
    voxel_size: Tuple[int, int, int],
    args: Namespace,
    executor: Executor,
) -> None:
    (
        min_paddings,
        min_dimensions,
        max_dimensions,
        arbitrary_file,
        file_count,
    ) = detect_interval_for_dimensions(input_path_pattern)

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

    target_ds = Dataset(target_path, voxel_size=voxel_size, exist_ok=True)
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

    target_mag_view = target_layer.get_or_add_mag(
        Mag(1), chunk_shape=DEFAULT_CHUNK_SHAPE.z
    )

    job_args = []
    # Iterate over all z batches
    for z_batch in get_regular_chunks(
        min_dimensions["z"], max_dimensions["z"], DEFAULT_CHUNK_SHAPE.z
    ):
        # The z_batch always starts and ends at a multiple of DEFAULT_CHUNK_SHAPE.z.
        # However, we only want the part that is inside the bounding box
        z_batch = range(
            max(list(z_batch)[0], target_layer.bounding_box.topleft.z),
            min(list(z_batch)[-1] + 1, target_layer.bounding_box.bottomright.z),
        )
        z_values = list(z_batch)
        job_args.append(
            (
                target_mag_view.get_view(
                    absolute_offset=(x_offset, y_offset, z_values[0]),
                    size=(num_x, num_y, len(z_values)),
                ),
                z_values,
                input_path_pattern,
                batch_size,
                (tile_width, tile_height, num_channels),
                min_dimensions,
                max_dimensions,
                min_paddings,
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
        cast(SegmentationLayer, target_layer).largest_segment_id = largest_segment_id


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
    setup_warnings()
    args = create_parser().parse_args()
    setup_logging(args)
    input_path_pattern = os.path.join(args.source_path, args.input_path_pattern)

    with get_executor_for_args(args) as executor:
        tile_cubing(
            args.target_path,
            args.layer_name,
            int(args.batch_size),
            input_path_pattern,
            args.voxel_size,
            args,
            executor=executor,
        )

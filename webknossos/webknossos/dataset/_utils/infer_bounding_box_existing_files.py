import os
import re
from typing import List, Tuple, Union

import numpy as np

from webknossos.dataset.mag_view import MagView
from webknossos.geometry import BoundingBox

WKW_CUBE_REGEX = re.compile(
    fr"z(\d+){re.escape(os.path.sep)}y(\d+){re.escape(os.path.sep)}x(\d+)(\.wkw)$"
)


def cube_addresses(mag_view: MagView) -> List[Tuple[int, int, int]]:
    # Gathers all WKW cubes in the dataset
    wkw_files = mag_view.path.glob("z*/y*/x*.wkw")
    return sorted(parse_cube_file_name(f) for f in wkw_files)


def parse_cube_file_name(filename: Union[os.PathLike, str]) -> Tuple[int, int, int]:
    match = WKW_CUBE_REGEX.search(str(filename))
    if match is None:
        raise ValueError(f"Failed to parse cube file name {filename}")
    return int(match.group(3)), int(match.group(2)), int(match.group(1))


def infer_bounding_box_existing_files(mag_view: MagView) -> BoundingBox:
    """Since volume tracings are only a single layer, they do not contain a datasource-properties.json.
    Therefore, the bounding box needs to be inferred when working with those."""

    # N x 3 array of cube addresses
    addresses = np.array(cube_addresses(mag_view))
    voxel_length = mag_view.header.file_len * mag_view.header.block_len

    top_left = addresses.min(axis=0) * voxel_length
    bottom_right = (addresses.max(axis=0) + 1) * voxel_length
    size = bottom_right - top_left

    return BoundingBox(top_left, size)

import random
from pathlib import Path

import numpy as np

from webknossos.dataset.mag_view import MagView
from webknossos.geometry.bounding_box import BoundingBox
from webknossos.geometry.vec3_int import Vec3Int

NUM_SAMPLES = 20
THRESHOLD = (
    1 / 200
)  # more unique values per voxel than this value means color, less segmentation
SAMPLE_SIZE = Vec3Int(16, 16, 16)
MAX_FAILS = 100


def guess_if_segmentation_path(filepath: Path) -> bool:
    lowercase_filepath = str(filepath).lower()
    return any(i in lowercase_filepath for i in ["segmentation", "labels"])


def guess_if_segmentation_from_view(view: MagView) -> bool:
    return sample_distinct_values_per_vx(view) <= THRESHOLD


def sample_distinct_values_per_vx(view: MagView) -> float:
    sample_size_for_view = view.size.pairmin(SAMPLE_SIZE * view.mag)
    min_offset = view.bounding_box.topleft
    max_offset = view.bounding_box.bottomright - sample_size_for_view

    distinct_color_values = 0
    valid_sample_count = 0
    inspected_voxel_count = 0
    invalid_sample_count = 0

    while valid_sample_count < NUM_SAMPLES:
        if invalid_sample_count > MAX_FAILS:
            raise RuntimeError("Failed to find enough valid samples.")
        offset = Vec3Int(
            random.randint(min_offset.x, max_offset.x),
            random.randint(min_offset.y, max_offset.y),
            random.randint(min_offset.z, max_offset.z),
        )
        bbox_to_read = BoundingBox(
            topleft=offset, size=sample_size_for_view
        ).align_with_mag(view.mag)
        data = view.read(absolute_bounding_box=bbox_to_read)
        size_before = data.size
        data = data[data != 0]
        data = data[data != np.iinfo(data.dtype).max]
        if data.size < size_before:
            print(f"size reduced from {size_before} to {data.size}")

        distinct_color_values_in_sample = np.unique(data)

        if len(distinct_color_values_in_sample) == 0:
            invalid_sample_count += 1
        else:
            distinct_color_values += len(distinct_color_values_in_sample)
            valid_sample_count += 1
            inspected_voxel_count += data.size

    return distinct_color_values / inspected_voxel_count

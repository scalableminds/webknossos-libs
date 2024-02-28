import random
from pathlib import Path

import numpy as np

from webknossos.dataset.view import View
from webknossos.geometry.vec3_int import Vec3Int

NUM_SAMPLES = 20
THRESHOLD = 1 / 32
SAMPLE_SIZE = Vec3Int(4, 4, 4)
MAX_FAILS = 100


def guess_if_segmentation_path(filepath: Path) -> bool:
    lowercase_filepath = str(filepath).lower()
    return any(i in lowercase_filepath for i in ["segmentation", "labels"])


def guess_if_segmentation_from_view(view: View) -> bool:
    min_x, min_y, min_z = view.bounding_box.topleft
    max_x, max_y, max_z = view.bounding_box.topleft + (
        view.bounding_box.size - SAMPLE_SIZE
    ).pairmin(Vec3Int(1, 1, 1))
    found_samples = 0
    color_values = 0
    fail_counter = 0

    while found_samples < NUM_SAMPLES:
        if fail_counter > MAX_FAILS:
            raise RuntimeError("Failed to find enough valid samples.")
        absolute_offset = Vec3Int(
            random.randint(min_x, max_x),
            random.randint(min_y, max_y),
            random.randint(min_z, max_z),
        )
        elements = np.unique(
            view.get_view(size=SAMPLE_SIZE, absolute_offset=absolute_offset).read()
        )
        if len(elements) == 1 and elements[0] == 0:
            fail_counter += 1

        color_values += len(elements)
        found_samples += 1

    return sample_distinct_values_per_vx(view) <= THRESHOLD


def sample_distinct_values_per_vx(view: View) -> float:
    min_x, min_y, min_z = view.bounding_box.topleft
    max_x, max_y, max_z = view.bounding_box.topleft + (
        view.bounding_box.size - SAMPLE_SIZE
    ).pairmin(Vec3Int(1, 1, 1))
    distinct_color_values = 0
    found_sample_count = 0
    inspected_voxel_count = 0
    invalid_sample_count = 0

    while found_sample_count < NUM_SAMPLES:
        if invalid_sample_count > MAX_FAILS:
            raise RuntimeError("Failed to find enough valid samples.")
        absolute_offset = Vec3Int(
            random.randint(min_x, max_x),
            random.randint(min_y, max_y),
            random.randint(min_z, max_z),
        )
        data = view.read(size=SAMPLE_SIZE, absolute_offset=absolute_offset)
        distinct_color_values_in_sample = np.unique(data)
        if (
            len(distinct_color_values_in_sample) == 1
            and distinct_color_values_in_sample[0] == 0
        ):
            invalid_sample_count += 1

        distinct_color_values += len(distinct_color_values_in_sample)
        found_sample_count += 1
        inspected_voxel_count += SAMPLE_SIZE.prod()

    return distinct_color_values / inspected_voxel_count

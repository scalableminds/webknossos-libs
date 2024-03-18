import random
from pathlib import Path

import numpy as np

from webknossos.dataset.layer_categories import LayerCategoryType
from webknossos.dataset.mag_view import MagView
from webknossos.geometry.bounding_box import BoundingBox
from webknossos.geometry.vec3_int import Vec3Int

NUM_SAMPLES = 20
THRESHOLD = (
    1 / 200
)  # more unique values per voxel than this value means color, less segmentation
SAMPLE_SIZE = Vec3Int(16, 16, 16)
MAX_FAILS = 200


def guess_if_segmentation_path(filepath: Path) -> bool:
    lowercase_filepath = str(filepath).lower()
    return any(i in lowercase_filepath for i in ["segmentation", "labels"])


def guess_category_from_view(view: MagView) -> LayerCategoryType:
    if view.layer.num_channels > 1 or sample_distinct_values_per_vx(view) > THRESHOLD:
        return "color"
    return "segmentation"


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
            break
        offset = Vec3Int(
            random.randint(min_offset.x, max_offset.x),
            random.randint(min_offset.y, max_offset.y),
            random.randint(min_offset.z, max_offset.z),
        )
        bbox_to_read = BoundingBox(
            topleft=offset, size=sample_size_for_view
        ).align_with_mag(view.mag)
        data = view.read(absolute_bounding_box=bbox_to_read)
        data = data[data != 0]
        try:
            data = data[data != np.iinfo(data.dtype).max]
        except Exception:  # pylint: disable=bare-except
            pass  # does not work or make sense for float data types

        distinct_color_values_in_sample = np.unique(data)

        if len(distinct_color_values_in_sample) == 0:
            invalid_sample_count += 1
        else:
            distinct_color_values += len(distinct_color_values_in_sample)
            valid_sample_count += 1
            inspected_voxel_count += data.size

    if inspected_voxel_count < 1000:
        raise RuntimeError(
            f"Failed to find enough valid samples (saw {inspected_voxel_count} voxels)."
        )

    return distinct_color_values / inspected_voxel_count

from functools import reduce

from ...geometry import BoundingBox
from ..mag_view import MagView


def infer_bounding_box_existing_files(mag_view: MagView) -> BoundingBox:
    """Since volume annotation layers are only a single layer, they do not contain a datasource-properties.json.
    Therefore, the bounding box needs to be inferred when working with those.
    The returned bounding box is measured in Mag(1) voxels."""

    return reduce(
        lambda acc, bbox: acc.extended_by(BoundingBox.from_ndbbox(bbox)),
        mag_view.get_bounding_boxes_on_disk(),
        BoundingBox.empty(),
    )

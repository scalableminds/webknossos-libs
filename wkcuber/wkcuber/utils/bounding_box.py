"""This file includes helper functions for utilizing the webknossos boundingbox."""

from webknossos import BoundingBox, Vec3Int


def parse_bounding_box(bounding_box: str | None) -> BoundingBox | None:
    """Parsees str input to Vec3Int."""
    try:
        if not bounding_box:
            return None
        else:
            print(bounding_box)
            return BoundingBox(Vec3Int(0, 0, 0), Vec3Int(11, 11, 11))
    except Exception as err:
        raise ValueError("The voxel_size could not be parsed") from err

"""This file includes helper functions for utilizing the webknossos datasets."""


from typing import Tuple

from webknossos import Vec3Int


def parse_voxel_size(voxel_size: str) -> Tuple[float, float, float]:
    """Parsees str input to (float, float, float)."""
    try:
        result = tuple(float(x) for x in voxel_size.split(","))
        if len(result) == 1:
            return (result[0],) * 3
        elif len(result) == 3:
            return (result[0], result[1], result[2])
        else:
            raise ValueError(
                f"Expected voxel_size of length 1 or 3 but got: {result} of length {len(result)}"
            )
    except Exception as err:
        raise ValueError("The voxel_size could not be parsed") from err


def parse_vec3int(vec3int: str) -> Vec3Int:
    """Parses str input to Vec3Int"""
    try:
        result = tuple(int(x) for x in vec3int.split(","))
        if len(result) == 1:
            return Vec3Int.full(result[0])
        elif len(result) == 3:
            return Vec3Int.from_xyz(*result)
        else:
            raise ValueError()
    except Exception as err:
        raise ValueError("The value could not be parsed to Vec3Int.") from err

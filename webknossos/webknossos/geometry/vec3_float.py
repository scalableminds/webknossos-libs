from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from .vec3_int import Vec3Int

Vec3Float: TypeAlias = tuple[float, float, float]
"""A three-dimensional vector of floats, e.g. a voxel size or a position with sub-voxel
precision. Use `Vec3Int` where only whole voxels are meaningful."""

Vec3FloatLike: TypeAlias = Vec3Float | Vec3Int | np.ndarray | Iterable[float]
"""Anything that `parse_vec3_float` can turn into a `Vec3Float`."""


def parse_vec3_float(vec3_float_like: Vec3FloatLike) -> Vec3Float:
    """Converts `vec3_float_like` into a `Vec3Float`.

    Accepts tuples, lists, numpy arrays, `Vec3Int` and any other iterable of three
    numbers.

    Raises:
        ValueError: If the value does not consist of exactly three numbers.
    """
    if not isinstance(vec3_float_like, np.ndarray):
        vec3_float_like = tuple(vec3_float_like)
    try:
        array = np.asarray(vec3_float_like, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"Vector components must be three floats, got {vec3_float_like!r}."
        ) from None
    if array.shape != (3,):
        raise ValueError(
            f"Vector components must be three floats, got an object of shape {array.shape}."
        )
    return (float(array[0]), float(array[1]), float(array[2]))

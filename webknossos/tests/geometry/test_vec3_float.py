import numpy as np
import pytest

from webknossos import Annotation, Skeleton, VoxelSize
from webknossos.geometry import (
    Vec3Int,
    parse_vec3_float,
    parse_vec3_float_or_none,
)


def test_parse_vec3_float() -> None:
    assert parse_vec3_float((1.5, 2, 3)) == (1.5, 2.0, 3.0)
    assert parse_vec3_float([1.5, 2, 3]) == (1.5, 2.0, 3.0)
    assert parse_vec3_float(np.array([1.5, 2.0, 3.0])) == (1.5, 2.0, 3.0)
    assert parse_vec3_float(Vec3Int(1, 2, 3)) == (1.0, 2.0, 3.0)
    assert parse_vec3_float(iter([1.5, 2, 3])) == (1.5, 2.0, 3.0)

    # Integers are widened to floats, the result is always a plain tuple of floats
    result = parse_vec3_float((1, 2, 3))
    assert result == (1.0, 2.0, 3.0)
    assert isinstance(result, tuple)
    assert all(isinstance(value, float) for value in result)


def test_parse_vec3_float_rejects_invalid_values() -> None:
    for invalid in [(1, 2), (1, 2, 3, 4), [], np.zeros((3, 1)), np.zeros((2, 3))]:
        with pytest.raises(ValueError, match="three floats"):
            parse_vec3_float(invalid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="three floats"):
        parse_vec3_float(("a", "b", "c"))  # type: ignore[arg-type]


def test_parse_vec3_float_or_none() -> None:
    assert parse_vec3_float_or_none(None) is None
    assert parse_vec3_float_or_none((1, 2, 3)) == (1.0, 2.0, 3.0)


def test_public_apis_accept_vec3_float_like() -> None:
    """The float-valued entry points coerce whatever `Vec3FloatLike` allows."""
    for value in [(1, 2, 3), [1, 2, 3], np.array([1.0, 2.0, 3.0]), Vec3Int(1, 2, 3)]:
        assert VoxelSize(value).factor == (1.0, 2.0, 3.0)  # type: ignore[arg-type]
    assert VoxelSize(iter([1, 2, 3])).factor == (1.0, 2.0, 3.0)  # type: ignore[arg-type]

    skeleton = Skeleton(voxel_size=(1, 1, 1), dataset_name="d")
    tree = skeleton.add_tree("t")
    node = tree.add_node(position=(0, 0, 0), rotation=np.array([1.0, 2.0, 3.0]))
    assert node.rotation == (1.0, 2.0, 3.0)

    skeleton.voxel_size = np.array([2.0, 2.0, 2.0])
    assert skeleton.voxel_size == (2.0, 2.0, 2.0)

    annotation = Annotation(
        name="a",
        dataset_name="d",
        voxel_size=(1, 1, 1),
        edit_position=np.array([1.0, 2.0, 3.0]),
    )
    assert annotation.edit_position == (1.0, 2.0, 3.0)

    # Malformed values are rejected at the boundary instead of being stored as-is
    with pytest.raises(ValueError, match="three floats"):
        tree.add_node(position=(1, 1, 1), rotation=(1, 2))  # type: ignore[arg-type]

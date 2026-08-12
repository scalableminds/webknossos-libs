import copy
import json
import pickle

import numpy as np
import pytest

from webknossos import Annotation, Skeleton, VoxelSize
from webknossos.dataset_properties import DatasetProperties, get_dataset_converter
from webknossos.geometry import Vec3Float, Vec3Int
from webknossos.geometry.vec3_float import as_vec3_float_or_none


def test_construction() -> None:
    expected = (1.5, 2.0, 3.0)
    assert Vec3Float(1.5, 2, 3) == expected
    assert Vec3Float((1.5, 2, 3)) == expected
    assert Vec3Float([1.5, 2, 3]) == expected
    assert Vec3Float(np.array([1.5, 2.0, 3.0])) == expected
    assert Vec3Float(iter([1.5, 2, 3])) == expected
    assert Vec3Float(Vec3Int(1, 2, 3)) == (1.0, 2.0, 3.0)
    assert Vec3Float.from_xyz(1.5, 2.0, 3.0) == expected
    assert Vec3Float.from_str("1.5,2,3") == expected
    assert Vec3Float.from_str("(1.5, 2, 3)") == expected
    assert Vec3Float.zeros() == (0.0, 0.0, 0.0)
    assert Vec3Float.ones() == (1.0, 1.0, 1.0)
    assert Vec3Float.full(2) == (2.0, 2.0, 2.0)

    # Integers are widened to floats
    assert all(isinstance(value, float) for value in Vec3Float(1, 2, 3))

    # An existing Vec3Float is passed through unchanged
    vector = Vec3Float(1.5, 2.0, 3.0)
    assert Vec3Float(vector) is vector

    assert as_vec3_float_or_none(None) is None
    assert as_vec3_float_or_none((1, 2, 3)) == (1.0, 2.0, 3.0)


def test_rejects_invalid_values() -> None:
    for invalid in [(1, 2), (1, 2, 3, 4), [], np.zeros((3, 1)), np.zeros((2, 3))]:
        with pytest.raises(ValueError, match="three floats"):
            Vec3Float(invalid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="three floats"):
        Vec3Float(("a", "b", "c"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="three floats"):
        Vec3Float(1, 2)  # type: ignore[call-overload]


def test_tuple_interoperability() -> None:
    """Instances must stay interchangeable with plain tuples, which is what keeps the
    existing `voxel_size == (1, 1, 1)` style assertions working."""
    vector = Vec3Float(1.0, 2.0, 3.0)

    assert vector == (1.0, 2.0, 3.0)
    assert (1.0, 2.0, 3.0) == vector
    assert vector != (1.0, 2.0, 4.0)

    assert hash(vector) == hash((1.0, 2.0, 3.0))
    assert vector in {(1.0, 2.0, 3.0)}
    assert {vector: "value"}[(1.0, 2.0, 3.0)] == "value"  # type: ignore[index]


def test_access_and_conversion() -> None:
    vector = Vec3Float(1.0, 2.0, 3.0)

    assert (vector.x, vector.y, vector.z) == (1.0, 2.0, 3.0)
    assert vector[0] == 1.0
    assert len(vector) == 3
    assert list(vector) == [1.0, 2.0, 3.0]
    assert 2.0 in vector

    assert vector.with_x(9) == (9.0, 2.0, 3.0)
    assert vector.with_y(9) == (1.0, 9.0, 3.0)
    assert vector.with_z(9) == (1.0, 2.0, 9.0)

    assert vector.to_tuple() == (1.0, 2.0, 3.0)
    assert vector.to_list() == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(vector.to_np(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(vector), np.array([1.0, 2.0, 3.0]))
    assert np.asarray(vector).dtype == np.float64

    assert vector.to_vec3_int() == Vec3Int(1, 2, 3)
    # Truncates towards zero rather than rounding
    assert Vec3Float(1.9, -1.9, 2.5).to_vec3_int() == Vec3Int(1, -1, 2)
    assert Vec3Float(Vec3Int(1, 2, 3).to_tuple()) == (1.0, 2.0, 3.0)

    assert repr(vector) == "Vec3Float(1.0,2.0,3.0)"


def test_arithmetic() -> None:
    vector = Vec3Float(1.0, 2.0, 3.0)

    # Element-wise, unlike plain tuples where + concatenates and * repeats
    assert vector + (1, 1, 1) == (2.0, 3.0, 4.0)
    assert vector + 1 == (2.0, 3.0, 4.0)
    assert vector - 1 == (0.0, 1.0, 2.0)
    assert vector * 2 == (2.0, 4.0, 6.0)
    assert vector * (1, 2, 3) == (1.0, 4.0, 9.0)
    assert vector / 2 == (0.5, 1.0, 1.5)
    assert -vector == (-1.0, -2.0, -3.0)

    assert vector.pairmax((2, 0, 9)) == (2.0, 2.0, 9.0)
    assert vector.pairmin((2, 0, 9)) == (1.0, 0.0, 3.0)
    assert vector.prod() == 6.0


def test_deepcopy_and_pickle() -> None:
    """Required because dataset properties and skeleton trees are deepcopied."""
    vector = Vec3Float(1.5, 2.0, 3.0)

    assert copy.deepcopy(vector) == vector
    assert copy.copy(vector) == vector

    restored = pickle.loads(pickle.dumps(vector))
    assert restored == vector
    assert isinstance(restored, Vec3Float)


def test_serialization_round_trip() -> None:
    """The headline risk: a class-typed cattrs field unstructures to identity
    pass-through unless hooks are registered, which makes json.dumps raise."""
    converter = get_dataset_converter()

    unstructured = converter.unstructure(VoxelSize((1.0, 1.0, 2.0)))
    assert unstructured == {"factor": [1.0, 1.0, 2.0], "unit": "nanometer"}
    # Must be a JSON array, not an object or a string
    assert json.loads(json.dumps(unstructured))["factor"] == [1.0, 1.0, 2.0]

    properties = converter.structure(
        {
            "id": {"name": "d", "team": "t"},
            "scale": {"factor": [1.0, 1.0, 2.0], "unit": "nanometer"},
            "dataLayers": [],
            "version": 1,
        },
        DatasetProperties,
    )
    assert isinstance(properties.scale.factor, Vec3Float)
    assert properties.scale.factor == (1.0, 1.0, 2.0)

    round_tripped = converter.structure(
        json.loads(json.dumps(converter.unstructure(properties))), DatasetProperties
    )
    assert round_tripped == properties


def test_public_apis_accept_vec3_float_like() -> None:
    """The float-valued entry points coerce whatever `Vec3FloatLike` allows."""
    for value in [(1, 2, 3), [1, 2, 3], np.array([1.0, 2.0, 3.0]), Vec3Int(1, 2, 3)]:
        assert VoxelSize(value).factor == (1.0, 2.0, 3.0)  # type: ignore[arg-type]
    assert VoxelSize(iter([1, 2, 3])).factor == (1.0, 2.0, 3.0)  # type: ignore[arg-type]
    assert isinstance(VoxelSize((1, 2, 3)).factor, Vec3Float)

    skeleton = Skeleton(voxel_size=(1, 1, 1), dataset_name="d")
    tree = skeleton.add_tree("t")
    node = tree.add_node(position=(0, 0, 0), rotation=np.array([1.0, 2.0, 3.0]))
    assert node.rotation == (1.0, 2.0, 3.0)
    assert isinstance(node.rotation, Vec3Float)

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


def test_tree_color_accepts_vec3_float() -> None:
    """A Vec3Float colour must be widened to RGBA, not element-wise added."""
    skeleton = Skeleton(voxel_size=(1, 1, 1), dataset_name="d")
    tree = skeleton.add_tree("t", color=Vec3Float(1.0, 0.0, 0.0))
    assert tree.color == (1.0, 0.0, 0.0, 1.0)


def test_ordering() -> None:
    """Vectors are ordered lexicographically, like the equivalent plain tuples.

    `Dataset.voxel_size` and friends returned plain tuples before `Vec3Float` became a
    class, so sorting and `min`/`max` over them has to keep working.
    """
    assert Vec3Float(1, 2, 3) < Vec3Float(1, 2, 4)
    assert Vec3Float(1, 2, 4) > Vec3Float(1, 2, 3)
    assert Vec3Float(1, 2, 3) <= Vec3Float(1, 2, 3)
    assert Vec3Float(1, 2, 3) >= Vec3Float(1, 2, 3)

    # Ordering against plain tuples works in both directions, as for `==`
    assert Vec3Float(1, 2, 3) < (1, 2, 4)
    assert (1, 2, 3) < Vec3Float(1, 2, 4)

    vectors = [Vec3Float(4, 4, 40), Vec3Float(1, 1, 10), Vec3Float(1, 1, 2)]
    assert sorted(vectors) == sorted(v.to_tuple() for v in vectors)
    assert min(vectors) == (1.0, 1.0, 2.0)

    with pytest.raises(TypeError):
        Vec3Float(1, 2, 3) < "abc"  # type: ignore[operator]

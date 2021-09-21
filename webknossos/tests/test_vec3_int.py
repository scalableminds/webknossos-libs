import numpy as np

from webknossos.geometry import Vec3Int


def test_import() -> None:

    assert Vec3Int(1, 2, 3) == Vec3Int(1, 2, 3)
    assert Vec3Int((1, 2, 3)) == Vec3Int(1, 2, 3)
    assert Vec3Int([1, 2, 3]) == Vec3Int(1, 2, 3)
    assert Vec3Int([1, 2, 3]) == Vec3Int(1, 2, 3)
    assert Vec3Int(np.array([1, 2, 3])) == Vec3Int(1, 2, 3)


def test_export() -> None:

    assert Vec3Int(1, 2, 3).x == 1
    assert Vec3Int(1, 2, 3).y == 2
    assert Vec3Int(1, 2, 3).z == 3
    assert Vec3Int(1, 2, 3)[0] == 1
    assert Vec3Int(1, 2, 3)[1] == 2
    assert Vec3Int(1, 2, 3)[2] == 3
    assert np.array_equal(Vec3Int(1, 2, 3).to_np(), np.array([1, 2, 3]))
    assert Vec3Int(1, 2, 3).to_list() == [1, 2, 3]
    assert Vec3Int(1, 2, 3).to_tuple() == (1, 2, 3)


def test_arithmetic() -> None:
    assert Vec3Int(1, 2, 3) + Vec3Int(4, 5, 6) == Vec3Int(5, 7, 9)
    assert Vec3Int(1, 2, 3) + Vec3Int(0, 0, 0) == Vec3Int(1, 2, 3)
    assert Vec3Int(1, 2, 3) - Vec3Int(4, 5, 6) == Vec3Int(-3, -3, -3)

    assert Vec3Int(1, 2, 3) * 3 == Vec3Int(3, 6, 9)
    assert Vec3Int(1, 2, 3) + 3 == Vec3Int(4, 5, 6)
    assert Vec3Int(1, 2, 3) - 3 == Vec3Int(-2, -1, 0)

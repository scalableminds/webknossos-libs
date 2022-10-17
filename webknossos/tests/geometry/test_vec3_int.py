import numpy as np

from webknossos.geometry import Mag, Vec3Int


def test_with() -> None:

    assert Vec3Int(1, 2, 3).with_x(5) == Vec3Int(5, 2, 3)
    assert Vec3Int(1, 2, 3).with_y(5) == Vec3Int(1, 5, 3)
    assert Vec3Int(1, 2, 3).with_z(5) == Vec3Int(1, 2, 5)


def test_import() -> None:

    assert Vec3Int(1, 2, 3) == Vec3Int(1, 2, 3)
    assert Vec3Int((1, 2, 3)) == Vec3Int(1, 2, 3)
    assert Vec3Int([1, 2, 3]) == Vec3Int(1, 2, 3)
    assert Vec3Int(i for i in [1, 2, 3]) == Vec3Int(1, 2, 3)
    assert Vec3Int(np.array([1, 2, 3])) == Vec3Int(1, 2, 3)
    assert Vec3Int(Mag(4)) == Vec3Int(4, 4, 4)


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


def test_operator_arithmetic() -> None:

    # other is Vec3Int
    assert Vec3Int(1, 2, 3) + Vec3Int(4, 5, 6) == Vec3Int(5, 7, 9)
    assert Vec3Int(1, 2, 3) + Vec3Int(0, 0, 0) == Vec3Int(1, 2, 3)
    assert Vec3Int(1, 2, 3) - Vec3Int(4, 5, 6) == Vec3Int(-3, -3, -3)
    assert Vec3Int(1, 2, 3) * Vec3Int(4, 5, 6) == Vec3Int(4, 10, 18)
    assert Vec3Int(4, 5, 6) // Vec3Int(1, 2, 3) == Vec3Int(4, 2, 2)
    assert Vec3Int(4, 5, 6) % Vec3Int(1, 2, 3) == Vec3Int(0, 1, 0)

    # other is scalar int
    assert Vec3Int(1, 2, 3) * 3 == Vec3Int(3, 6, 9)
    assert Vec3Int(1, 2, 3) + 3 == Vec3Int(4, 5, 6)
    assert Vec3Int(1, 2, 3) - 3 == Vec3Int(-2, -1, 0)
    assert Vec3Int(4, 5, 6) // 2 == Vec3Int(2, 2, 3)
    assert Vec3Int(4, 5, 6) % 3 == Vec3Int(1, 2, 0)

    # other is Vec3IntLike (e.g. tuple)
    assert Vec3Int(1, 2, 3) + (4, 5, 6) == Vec3Int(5, 7, 9)

    # be wary of the tuple “+” operation:
    assert (1, 2, 3) + Vec3Int(4, 5, 6) == (1, 2, 3, 4, 5, 6)

    assert -Vec3Int(1, 2, 3) == Vec3Int(-1, -2, -3)


def test_method_arithmetic() -> None:

    assert Vec3Int(4, 5, 6).ceildiv(Vec3Int(1, 2, 3)) == Vec3Int(4, 3, 2)
    assert Vec3Int(4, 5, 6).ceildiv((1, 2, 3)) == Vec3Int(4, 3, 2)
    assert Vec3Int(4, 5, 6).ceildiv(2) == Vec3Int(2, 3, 3)

    assert Vec3Int(1, 2, 6).pairmax(Vec3Int(4, 5, 3)) == Vec3Int(4, 5, 6)
    assert Vec3Int(1, 2, 6).pairmin(Vec3Int(4, 5, 3)) == Vec3Int(1, 2, 3)


def test_repr() -> None:

    assert str(Vec3Int(1, 2, 3)) == "Vec3Int(1,2,3)"


def test_prod() -> None:

    assert Vec3Int(1, 2, 3).prod() == 6


def test_contains() -> None:

    assert Vec3Int(1, 2, 3).contains(1)
    assert not Vec3Int(1, 2, 3).contains(4)


def test_custom_initialization() -> None:

    assert Vec3Int.zeros() == Vec3Int(0, 0, 0)
    assert Vec3Int.ones() == Vec3Int(1, 1, 1)
    assert Vec3Int.full(4) == Vec3Int(4, 4, 4)

    assert Vec3Int.ones() - Vec3Int.ones() == Vec3Int.zeros()
    assert Vec3Int.full(4) == Vec3Int.ones() * 4

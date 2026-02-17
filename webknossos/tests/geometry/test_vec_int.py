import numpy as np

from webknossos.geometry import Mag, VecInt


def test_with_replaced() -> None:
    assert VecInt(x=1, y=2, z=3, t=4).with_replaced(0, 5) == VecInt(x=5, y=2, z=3, t=4)
    assert VecInt(x=1, y=2, z=3, t=4).with_replaced(1, 5) == VecInt(x=1, y=5, z=3, t=4)
    assert VecInt(x=1, y=2, z=3, t=4).with_replaced(2, 5) == VecInt(x=1, y=2, z=5, t=4)
    assert VecInt(x=1, y=2, z=3, t=4).with_replaced(3, 5) == VecInt(x=1, y=2, z=3, t=5)


def test_import() -> None:
    assert VecInt(x=1, y=2, z=3, t=4) == VecInt(x=1, y=2, z=3, t=4)
    assert VecInt(1, 2, 3, 4, axes=("x", "y", "z", "t")) == VecInt(x=1, y=2, z=3, t=4)
    assert VecInt((1, 2, 3, 4), axes=("x", "y", "z", "t")) == VecInt(x=1, y=2, z=3, t=4)
    assert VecInt([1, 2, 3, 4], axes=("x", "y", "z", "t")) == VecInt(x=1, y=2, z=3, t=4)
    assert VecInt((i for i in (1, 2, 3, 4)), axes=("x", "y", "z", "t")) == VecInt(
        x=1, y=2, z=3, t=4
    )
    assert VecInt(np.array([1, 2, 3, 4]), axes=("x", "y", "z", "t")) == VecInt(
        x=1, y=2, z=3, t=4
    )
    assert VecInt(Mag(4), axes=("x", "y", "z")) == VecInt(x=4, y=4, z=4)


def test_export() -> None:
    assert VecInt(x=1, y=2, z=3, t=4).x == 1
    assert VecInt(x=1, y=2, z=3, t=4).y == 2
    assert VecInt(x=1, y=2, z=3, t=4).z == 3
    assert VecInt(x=1, y=2, z=3, t=4)[0] == 1
    assert VecInt(x=1, y=2, z=3, t=4)[1] == 2
    assert VecInt(x=1, y=2, z=3, t=4)[2] == 3
    assert np.array_equal(VecInt(x=1, y=2, z=3, t=4).to_np(), np.array([1, 2, 3, 4]))
    assert VecInt(x=1, y=2, z=3, t=4).to_list() == [1, 2, 3, 4]
    assert VecInt(x=1, y=2, z=3, t=4).to_tuple() == (1, 2, 3, 4)


def test_operator_arithmetic() -> None:
    # other is VecInt
    assert VecInt(x=1, y=2, z=3, t=4) + VecInt(x=5, y=6, z=7, t=8) == VecInt(
        x=6, y=8, z=10, t=12
    )
    assert VecInt(x=1, y=2, z=3, t=4) + VecInt(x=0, y=0, z=0, t=0) == VecInt(
        x=1, y=2, z=3, t=4
    )
    assert VecInt(x=1, y=2, z=3, t=4) - VecInt(x=5, y=6, z=7, t=8) == VecInt(
        x=-4, y=-4, z=-4, t=-4
    )
    assert VecInt(x=1, y=2, z=3, t=4) * VecInt(x=5, y=6, z=7, t=8) == VecInt(
        x=5, y=12, z=21, t=32
    )
    assert VecInt(x=5, y=6, z=7, t=8) // VecInt(x=1, y=2, z=3, t=4) == VecInt(
        x=5, y=3, z=2, t=2
    )
    assert VecInt(x=5, y=6, z=7, t=8) % VecInt(x=1, y=2, z=3, t=4) == VecInt(
        x=0, y=0, z=1, t=0
    )

    # other is scalar int
    assert VecInt(x=1, y=2, z=3, t=4) * 3 == VecInt(x=3, y=6, z=9, t=12)
    assert VecInt(x=1, y=2, z=3, t=4) + 3 == VecInt(x=4, y=5, z=6, t=7)
    assert VecInt(x=1, y=2, z=3, t=4) - 3 == VecInt(x=-2, y=-1, z=0, t=1)
    assert VecInt(x=5, y=6, z=7, t=8) // 2 == VecInt(x=2, y=3, z=3, t=4)
    assert VecInt(x=5, y=6, z=7, t=8) % 3 == VecInt(x=2, y=0, z=1, t=2)

    # other is VecIntLike (e.g. tuple)
    assert VecInt(x=1, y=2, z=3, t=4) + (4, 5, 6, 7) == VecInt(x=5, y=7, z=9, t=11)

    assert -VecInt(x=1, y=2, z=3, t=4) == VecInt(x=-1, y=-2, z=-3, t=-4)


def test_method_arithmetic() -> None:
    assert VecInt(x=5, y=6, z=7, t=8).ceildiv(VecInt(x=1, y=2, z=3, t=4)) == VecInt(
        x=5, y=3, z=3, t=2
    )
    assert VecInt(x=5, y=6, z=7, t=8).ceildiv((1, 2, 3, 4)) == VecInt(
        x=5, y=3, z=3, t=2
    )
    assert VecInt(x=5, y=6, z=7, t=8).ceildiv(2) == VecInt(x=3, y=3, z=4, t=4)

    assert VecInt(x=1, y=2, z=7, t=2).pairmax(VecInt(x=4, y=5, z=3, t=4)) == VecInt(
        x=4, y=5, z=7, t=4
    )
    assert VecInt(x=1, y=2, z=6, t=3).pairmin(VecInt(x=4, y=5, z=3, t=4)) == VecInt(
        x=1, y=2, z=3, t=3
    )


def test_repr() -> None:
    assert str(VecInt(1, 2, axes=("x", "y"))) == "VecInt(1,2, axes=('x', 'y'))"


def test_prod() -> None:
    assert VecInt(x=1, y=2, z=3, t=4).prod() == 24


def test_contains() -> None:
    assert VecInt(x=1, y=2, z=3, t=4).contains(1)
    assert not VecInt(x=1, y=2, z=3, t=4).contains(5)


def test_custom_initialization() -> None:
    assert VecInt.zeros(("x", "y", "z", "t")) == VecInt(x=0, y=0, z=0, t=0)
    assert VecInt.ones(("x", "y", "z", "t")) == VecInt(x=1, y=1, z=1, t=1)
    assert VecInt.full(4, ("x", "y", "z", "t")) == VecInt(x=4, y=4, z=4, t=4)

    assert VecInt.ones(("x", "y", "z", "t")) - VecInt.ones(
        ("x", "y", "z", "t")
    ) == VecInt.zeros(("x", "y", "z", "t"))
    assert VecInt.full(4, ("x", "y", "z", "t")) == VecInt.ones(("x", "y", "z", "t")) * 4


def test_new_axes() -> None:
    old = VecInt(1, 2, 3, 4, axes=("unset_0", "unset_1", "unset_2", "unset_3"))
    new = VecInt(1, 2, 3, 4, axes=("x", "y", "z", "t"))
    assert VecInt(old, axes=("x", "y", "z", "t")) == new

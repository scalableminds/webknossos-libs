import numpy as np
import pytest
from hypothesis import given, infer

from webknossos.geometry import Mag, NDBoundingBox, VecInt


def test_align_with_mag_ceiled() -> None:
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (10, 10, 10, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2), ceil=True) == NDBoundingBox(
        (0, 0, 0, 1, 1),
        (12, 12, 12, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2), ceil=True) == NDBoundingBox(
        (0, 0, 0, 1, 1),
        (10, 10, 10, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(4), ceil=True) == NDBoundingBox(
        (0, 0, 0, 1, 1),
        (12, 12, 12, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2), ceil=True) == NDBoundingBox(
        (0, 2, 2, 4, 5),
        (10, 10, 10, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )


def test_xyz_properties() -> None:
    bb = NDBoundingBox(
        (1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert bb.size_xyz == VecInt(x=6, y=7, z=8)
    assert bb.topleft_xyz == VecInt(x=1, y=2, z=3)
    assert bb.bottomright_xyz == VecInt(x=7, y=9, z=11)
    assert bb.index_xyz == VecInt(x=1, y=2, z=3)
    bb2 = NDBoundingBox(
        (1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10),
        ("z", "t", "s", "x", "y"),
        (1, 2, 3, 4, 5),
    )
    assert bb2.size_xyz == VecInt(x=9, y=10, z=6)
    assert bb2.topleft_xyz == VecInt(x=4, y=5, z=1)
    assert bb2.bottomright_xyz == VecInt(x=13, y=15, z=7)
    assert bb2.index_xyz == VecInt(x=4, y=5, z=1)


def test_xyz_methods() -> None:
    bb = NDBoundingBox(
        (1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert bb.with_size_xyz((10, 11, 12)) == NDBoundingBox(
        (1, 2, 3, 4, 5),
        (10, 11, 12, 9, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert bb.with_topleft_xyz((10, 11, 12)) == NDBoundingBox(
        (10, 11, 12, 4, 5),
        (6, 7, 8, 9, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert bb.with_bottomright_xyz((10, 11, 12)) == NDBoundingBox(
        (1, 2, 3, 4, 5),
        (9, 9, 9, 9, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert bb.with_index_xyz((3, 2, 1)) == NDBoundingBox(
        (3, 2, 1, 4, 5),
        (8, 7, 6, 9, 10),
        ("z", "y", "x", "t", "s"),
        (1, 2, 3, 4, 5),
    )


def test_align_with_mag_floored() -> None:
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (10, 10, 10, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2)) == NDBoundingBox(
        (2, 2, 2, 1, 1),
        (8, 8, 8, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2)) == NDBoundingBox(
        (2, 2, 2, 1, 1),
        (8, 8, 8, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(4)) == NDBoundingBox(
        (4, 4, 4, 1, 1),
        (4, 4, 4, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (9, 9, 9, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).align_with_mag(Mag(2)) == NDBoundingBox(
        (2, 2, 4, 4, 5),
        (8, 8, 8, 9, 9),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )


def test_in_mag() -> None:
    with pytest.raises(AssertionError):
        NDBoundingBox(
            (1, 2, 3, 4, 5),
            (9, 9, 9, 9, 9),
            ("x", "y", "z", "t", "s"),
            (1, 2, 3, 4, 5),
        ).in_mag(Mag(2))

    with pytest.raises(AssertionError):
        NDBoundingBox(
            (2, 2, 2, 2, 2),
            (9, 9, 9, 9, 9),
            ("x", "y", "z", "t", "s"),
            (1, 2, 3, 4, 5),
        ).in_mag(Mag(2))

    assert NDBoundingBox(
        (2, 2, 2, 2, 2),
        (10, 10, 10, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).in_mag(Mag(2)) == NDBoundingBox(
        (1, 1, 1, 2, 2),
        (5, 5, 5, 10, 10),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )


def test_with_bounds() -> None:
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).with_bounds("x", 0, 10) == NDBoundingBox(
        (0, 2, 3, 4, 5),
        (10, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).with_bounds("y", new_topleft=0) == NDBoundingBox(
        (1, 0, 3, 4, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).with_bounds("z", new_size=10) == NDBoundingBox(
        (1, 2, 3, 4, 5),
        (5, 5, 10, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (1, 2, 3, 4, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).with_bounds("t", new_size=-4) == NDBoundingBox(
        (1, 2, 3, 0, 5),
        (5, 5, 5, 4, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )


def test_contains() -> None:
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(VecInt((1, 1, 1, 1, 1), ("x", "y", "z", "t", "s")))
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(VecInt((5, 5, 5, 5, 5), ("x", "y", "z", "t", "s")))
    assert not NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(VecInt((6, 6, 6, 6, 6), ("x", "y", "z", "t", "s")))
    assert not NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(VecInt((20, 20, 20, 20, 20), ("x", "y", "z", "t", "s")))
    assert NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(np.array([5.5, 5.5, 5.5, 5.5, 5.5]))
    assert not NDBoundingBox(
        (1, 1, 1, 1, 1),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ).contains(np.array([6.0, 6.0, 6.0, 6.0, 6.0]))


@given(bb=infer, mag=infer, ceil=infer)
def test_align_with_mag_against_numpy_implementation(
    bb: NDBoundingBox,
    mag: Mag,
    ceil: bool,
) -> None:
    try:
        slow_np_result = bb._align_with_mag_slow(mag, ceil)
    # Very large numbers don't fit into the C-int anymore:
    except OverflowError:
        bb.align_with_mag(mag, ceil)
    else:
        # The slower numpy implementation is wrong for very large numbers:
        # Floating point precision for 64 bit floats is not capable of representing
        # numbers larger than 2**53 accurately.
        if all(i < 2**53 for i in bb.bottomright):
            assert bb.align_with_mag(mag, ceil) == slow_np_result


def test_negative_size() -> None:
    assert NDBoundingBox(
        (10, 10, 10, 10, 10),
        (-5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ) == NDBoundingBox(
        (5, 10, 10, 10, 10),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (10, 10, 10, 10, 10),
        (-5, 5, -5, 5, -5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ) == NDBoundingBox(
        (5, 10, 5, 10, 5),
        (5, 5, 5, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )
    assert NDBoundingBox(
        (10, 10, 10, 10, 10),
        (-5, 5, -50, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    ) == NDBoundingBox(
        (5, 10, -40, 10, 10),
        (5, 5, 50, 5, 5),
        ("x", "y", "z", "t", "s"),
        (1, 2, 3, 4, 5),
    )


@given(bbox=infer)
def test_negative_inversion(
    bbox: NDBoundingBox,
) -> None:
    """Flipping the topleft and bottomright (by padding both with the negative size)
    results in the original bbox, as negative sizes are converted to positive ones."""
    assert bbox == bbox.padded_with_margins(-bbox.size, -bbox.size)

import numpy as np
import pytest
from hypothesis import given, infer

from webknossos import Mag, NDBoundingBox


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


# def test_contains() -> None:
#     assert NDBoundingBox((1, 1, 1), (5, 5, 5)).contains((2, 2, 2))
#     assert NDBoundingBox((1, 1, 1), (5, 5, 5)).contains((1, 1, 1))

#     # top-left is inclusive, bottom-right is exclusive
#     assert not NDBoundingBox((1, 1, 1), (5, 5, 5)).contains((6, 6, 6))
#     assert not NDBoundingBox((1, 1, 1), (5, 5, 5)).contains((20, 20, 20))

#     # nd-array may contain float values
#     assert NDBoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([5.5, 5.5, 5.5]))
#     assert not NDBoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([6.0, 6.0, 6.0]))


# @given(bb=infer, mag=infer, ceil=infer)
# def test_align_with_mag_against_numpy_implementation(
#     bb: NDBoundingBox,
#     mag: Mag,
#     ceil: bool,
# ) -> None:
#     try:
#         slow_np_result = bb._align_with_mag_slow(mag, ceil)
#     # Very large numbers don't fit into the C-int anymore:
#     except OverflowError:
#         bb.align_with_mag(mag, ceil)
#     else:
#         # The slower numpy implementation is wrong for very large numbers:
#         # Floating point precision for 64 bit floats is not capable of representing
#         # numbers larger than 2**53 accurately.
#         if all(i < 2**53 for i in bb.bottomright):
#             assert bb.align_with_mag(mag, ceil) == slow_np_result


# def test_negative_size() -> None:
#     assert NDBoundingBox((10, 10, 10), (-5, 5, 5)) == NDBoundingBox(
#         (5, 10, 10), (5, 5, 5)
#     )
#     assert NDBoundingBox((10, 10, 10), (-5, 5, -5)) == NDBoundingBox(
#         (5, 10, 5), (5, 5, 5)
#     )
#     assert NDBoundingBox((10, 10, 10), (-5, 5, -50)) == NDBoundingBox(
#         (5, 10, -40), (5, 5, 50)
#     )


# @given(bbox=infer)
# def test_negative_inversion(
#     bbox: NDBoundingBox,
# ) -> None:
#     """Flipping the topleft and bottomright (by padding both with the negative size)
#     results in the original bbox, as negative sizes are converted to positive ones."""
#     assert bbox == bbox.padded_with_margins(-bbox.size, -bbox.size)

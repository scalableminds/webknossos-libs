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


def test_iter_chunk_starts_nd() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(20, 20, 20, 3, 2),
        axes=("x", "y", "z", "c", "t"),
    )
    starts = list(bbox.iter_chunk_starts((16, 16, 16)))
    assert all(type(v) is int for start in starts for v in start)
    # An xyz chunk shape chunks xyz, spans the whole c axis and iterates all other axes
    # one index at a time.
    assert starts == [
        (x, y, z, 0, t)
        for x in (0, 16)
        for y in (0, 16)
        for z in (0, 16)
        for t in (0, 1)
    ]


def test_iter_chunk_starts_three_entries_are_ambiguous_without_z_axis() -> None:
    bbox = NDBoundingBox(
        topleft=(10, 20, 5),
        size=(30, 30, 30),
        axes=("x", "y", "t"),
    )
    # Three entries for a three-axis box that is not xyz could be either reading, so
    # an unnamed shape is rejected instead of silently dropping the third entry.
    with pytest.raises(ValueError, match="ambiguous"):
        list(bbox.iter_chunk_starts((16, 16, 16)))

    # A named shape says which entry belongs to which axis and is used verbatim.
    assert list(bbox.iter_chunk_starts(VecInt(16, 16, 16, axes=("x", "y", "t")))) == [
        (x, y, t) for x in (10, 26) for y in (20, 36) for t in (5, 21)
    ]


def test_iter_chunk_starts_xyz_shorthand_maps_by_axis_name() -> None:
    """Three unnamed entries are the sizes of x, y and z whatever the axis order is."""
    bbox = NDBoundingBox(topleft=(0, 0, 0), size=(64, 64, 64), axes=("z", "y", "x"))
    assert [tuple(chunk.size) for chunk in bbox.chunk((8, 16, 32))][0] == (32, 16, 8)
    assert list(bbox.iter_chunk_starts((8, 16, 32)))[:2] == [(0, 0, 0), (0, 0, 8)]


def test_iter_chunk_starts_named_shape_is_reordered_to_box_axes() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0),
        size=(20, 20, 20, 4),
        axes=("x", "y", "z", "t"),
    )
    reordered = VecInt(2, 16, 16, 16, axes=("t", "z", "y", "x"))
    assert list(bbox.iter_chunk_starts(reordered)) == list(
        bbox.iter_chunk_starts(VecInt(16, 16, 16, 2, axes=bbox.axes))
    )


def test_iter_chunk_starts_xyz_shorthand_ignores_missing_z_axis() -> None:
    """With more than three axes a three-entry shape is unambiguous, so the entry for
    the missing z axis is simply unused."""
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0),
        size=(20, 20, 3, 2),
        axes=("x", "y", "t", "s"),
    )
    assert list(bbox.iter_chunk_starts((16, 16, 16))) == [
        (x, y, t, s)
        for x in (0, 16)
        for y in (0, 16)
        for t in (0, 1, 2)
        for s in (0, 1)
    ]


def test_iter_chunk_starts_full_length_shape() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(20, 20, 20, 3, 4),
        axes=("x", "y", "z", "c", "t"),
    )
    # A chunk shape with one entry per axis is used verbatim, i.e. the c axis is not
    # forced to span all channels.
    assert list(bbox.iter_chunk_starts((16, 16, 16, 2, 2))) == [
        (x, y, z, c, t)
        for x in (0, 16)
        for y in (0, 16)
        for z in (0, 16)
        for c in (0, 2)
        for t in (0, 2)
    ]


def test_chunk_nd() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(20, 20, 20, 3, 2),
        axes=("x", "y", "z", "c", "t"),
    )
    axes = ("x", "y", "z", "c", "t")
    # Border chunks are clipped to the box, the c axis is never split.
    assert list(bbox.chunk((16, 16, 16))) == [
        NDBoundingBox(
            topleft=VecInt(x, y, z, 0, t, axes=axes),
            size=VecInt(x_size, y_size, z_size, 3, 1, axes=axes),
            axes=axes,
        )
        for x, x_size in ((0, 16), (16, 4))
        for y, y_size in ((0, 16), (16, 4))
        for z, z_size in ((0, 16), (16, 4))
        for t in (0, 1)
    ]


def test_iter_chunk_starts_alignment_divisibility_assertion_nd() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(10, 10, 10, 1, 1),
        axes=("x", "y", "z", "c", "t"),
    )
    with pytest.raises(AssertionError):
        list(bbox.iter_chunk_starts((32, 32, 32), (7, 7, 7)))


def test_iter_overlapping_grid_cells_nd_clip_to() -> None:
    bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(50, 60, 70, 3, 4),
        axes=("x", "y", "z", "c", "t"),
    )
    clip = NDBoundingBox(
        topleft=(0, 0, 0, 0, 0),
        size=(20, 20, 20, 3, 4),
        axes=("x", "y", "z", "c", "t"),
    )
    cells = list(bbox.iter_overlapping_grid_cells((16, 16, 16), clip_to=clip))
    assert all(type(v) is int for cell in cells for v in cell)
    # xyz overlap cells 0 and 16; all non-xyz axes (c and t) use a cell size of 1, so
    # every index in the clip range is yielded for them.
    assert cells == [
        (x, y, z, c, t)
        for x in (0, 16)
        for y in (0, 16)
        for z in (0, 16)
        for c in (0, 1, 2)
        for t in (0, 1, 2, 3)
    ]


def test_iter_overlapping_grid_cells_nd_grid_is_independent_of_box_size() -> None:
    """Unlike `chunk()`, the origin-aligned grid must not depend on the box's own
    size on the channel axis."""
    cells_per_c_size = [
        list(
            NDBoundingBox(
                topleft=(0, 0, 0, 2),
                size=(8, 8, 8, c_size),
                axes=("x", "y", "z", "c"),
            ).iter_overlapping_grid_cells((8, 8, 8))
        )
        for c_size in (1, 2)
    ]
    assert cells_per_c_size[0] == [(0, 0, 0, 2)]
    assert cells_per_c_size[1] == [(0, 0, 0, 2), (0, 0, 0, 3)]


def test_iter_chunk_starts_nd_alignment_and_negative_coordinates() -> None:
    bbox = NDBoundingBox(
        topleft=(-33, 7, -64, 0, 0),
        size=(50, 60, 70, 3, 2),
        axes=("x", "y", "z", "c", "t"),
    )
    starts = list(bbox.iter_chunk_starts((32, 32, 32), (16, 16, 16)))
    assert all(type(v) is int for start in starts for v in start)

    # Per axis the first start is the previous multiple of the alignment:
    # -33 % 16 == 15 -> -48, 7 % 16 == 7 -> 0, -64 % 16 == 0 -> -64.
    assert starts == [
        (x, y, z, 0, t)
        for x in (-48, -16, 16)
        for y in (0, 32, 64)
        for z in (-64, -32, 0)
        for t in (0, 1)
    ]

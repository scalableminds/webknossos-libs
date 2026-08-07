import numpy as np
import pytest
from hypothesis import given, infer

from webknossos import BoundingBox, Mag
from webknossos.geometry import NormalizedBoundingBox


def test_align_with_mag_ceiled() -> None:
    assert BoundingBox((1, 1, 1), (10, 10, 10)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(12, 12, 12))
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(10, 10, 10))
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(
        Mag(4), ceil=True
    ) == BoundingBox(topleft=(0, 0, 0), size=(12, 12, 12))
    assert BoundingBox((1, 2, 3), (9, 9, 9)).align_with_mag(
        Mag(2), ceil=True
    ) == BoundingBox(topleft=(0, 2, 2), size=(10, 10, 10))


def test_align_with_mag_floored() -> None:
    assert BoundingBox((1, 1, 1), (10, 10, 10)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 2), size=(8, 8, 8)
    )
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 2), size=(8, 8, 8)
    )
    assert BoundingBox((1, 1, 1), (9, 9, 9)).align_with_mag(Mag(4)) == BoundingBox(
        topleft=(4, 4, 4), size=(4, 4, 4)
    )
    assert BoundingBox((1, 2, 3), (9, 9, 9)).align_with_mag(Mag(2)) == BoundingBox(
        topleft=(2, 2, 4), size=(8, 8, 8)
    )


def test_in_mag() -> None:
    with pytest.raises(AssertionError):
        BoundingBox((1, 2, 3), (9, 9, 9)).in_mag(Mag(2))

    with pytest.raises(AssertionError):
        BoundingBox((2, 2, 2), (9, 9, 9)).in_mag(Mag(2))

    assert BoundingBox((2, 2, 2), (10, 10, 10)).in_mag(Mag(2)) == BoundingBox(
        topleft=(1, 1, 1), size=(5, 5, 5)
    )


def test_with_bounds() -> None:
    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_x(0, 10) == BoundingBox(
        (0, 2, 3), (10, 5, 5)
    )
    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_y(
        new_topleft_y=0
    ) == BoundingBox((1, 0, 3), (5, 5, 5))
    assert BoundingBox((1, 2, 3), (5, 5, 5)).with_bounds_z(
        new_size_z=10
    ) == BoundingBox((1, 2, 3), (5, 5, 10))


def test_contains() -> None:
    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains((2, 2, 2))
    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains((1, 1, 1))

    # top-left is inclusive, bottom-right is exclusive
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains((6, 6, 6))
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains((20, 20, 20))

    # nd-array may contain float values
    assert BoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([5.5, 5.5, 5.5]))
    assert not BoundingBox((1, 1, 1), (5, 5, 5)).contains(np.array([6.0, 6.0, 6.0]))


@given(bb=infer, mag=infer, ceil=infer)
def test_align_with_mag_against_numpy_implementation(
    bb: BoundingBox,
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
    assert BoundingBox((10, 10, 10), (-5, 5, 5)) == BoundingBox((5, 10, 10), (5, 5, 5))
    assert BoundingBox((10, 10, 10), (-5, 5, -5)) == BoundingBox((5, 10, 5), (5, 5, 5))
    assert BoundingBox((10, 10, 10), (-5, 5, -50)) == BoundingBox(
        (5, 10, -40), (5, 5, 50)
    )


@given(bbox=infer)
def test_negative_inversion(
    bbox: BoundingBox,
) -> None:
    """Flipping the topleft and bottomright (by padding both with the negative size)
    results in the original bbox, as negative sizes are converted to positive ones."""
    assert bbox == bbox.padded_with_margins(-bbox.size, -bbox.size)


def test_eq_with_normalized_bbox() -> None:
    """Test equality between BoundingBox and NormalizedBoundingBox ignoring channel axis."""
    bbox = BoundingBox((1, 2, 3), (4, 5, 6))
    normalized = bbox.normalize_axes(3)

    # Both directions should work
    assert bbox == normalized
    assert normalized == bbox

    # Different bboxes should not be equal
    other_bbox = BoundingBox((0, 0, 0), (1, 1, 1))
    other_normalized = other_bbox.normalize_axes(2)
    assert bbox != other_normalized
    assert normalized != other_bbox


def test_intersected_with_normalized_bbox() -> None:
    """Test intersection between BoundingBox and NormalizedBoundingBox."""
    bbox1 = BoundingBox((0, 0, 0), (10, 10, 10))
    bbox2 = BoundingBox((5, 5, 5), (10, 10, 10))
    normalized1 = bbox1.normalize_axes(3)
    normalized2 = bbox2.normalize_axes(2)

    expected = BoundingBox((5, 5, 5), (5, 5, 5))

    # BoundingBox.intersected_with(NormalizedBoundingBox)
    assert bbox1.intersected_with(normalized2) == expected
    assert bbox2.intersected_with(normalized1) == expected

    # NormalizedBoundingBox.intersected_with(BoundingBox)
    assert normalized1.intersected_with(bbox2) == expected
    assert normalized2.intersected_with(bbox1) == expected

    # Result type should be NormalizedBoundingBox with correct channel count
    result1 = bbox1.intersected_with(normalized2)
    assert isinstance(result1, NormalizedBoundingBox)
    assert result1.size.c == 2

    result2 = normalized1.intersected_with(bbox2)
    assert isinstance(result2, NormalizedBoundingBox)
    assert result2.size.c == 3


def test_extended_by_normalized_bbox() -> None:
    """Test extension between BoundingBox and NormalizedBoundingBox."""
    bbox1 = BoundingBox((0, 0, 0), (5, 5, 5))
    bbox2 = BoundingBox((10, 10, 10), (5, 5, 5))
    normalized1 = bbox1.normalize_axes(3)
    normalized2 = bbox2.normalize_axes(2)

    expected = BoundingBox((0, 0, 0), (15, 15, 15))

    # BoundingBox.extended_by(NormalizedBoundingBox)
    assert bbox1.extended_by(normalized2) == expected
    assert bbox2.extended_by(normalized1) == expected

    # NormalizedBoundingBox.extended_by(BoundingBox)
    assert normalized1.extended_by(bbox2) == expected
    assert normalized2.extended_by(bbox1) == expected

    # Result type should be NormalizedBoundingBox with correct channel count
    result1 = bbox1.extended_by(normalized2)
    assert isinstance(result1, NormalizedBoundingBox)
    assert result1.size.c == 2

    result2 = normalized1.extended_by(bbox2)
    assert isinstance(result2, NormalizedBoundingBox)
    assert result2.size.c == 3


def test_contains_bbox_with_normalized_bbox() -> None:
    """Test contains_bbox between BoundingBox and NormalizedBoundingBox."""
    outer = BoundingBox((0, 0, 0), (10, 10, 10))
    inner = BoundingBox((2, 2, 2), (3, 3, 3))
    not_contained = BoundingBox((8, 8, 8), (5, 5, 5))

    outer_normalized = outer.normalize_axes(3)
    inner_normalized = inner.normalize_axes(2)
    not_contained_normalized = not_contained.normalize_axes(1)

    # BoundingBox.contains_bbox(NormalizedBoundingBox)
    assert outer.contains_bbox(inner_normalized)
    assert not outer.contains_bbox(not_contained_normalized)

    # NormalizedBoundingBox.contains_bbox(BoundingBox)
    assert outer_normalized.contains_bbox(inner)
    assert not outer_normalized.contains_bbox(not_contained)


@pytest.mark.parametrize(
    "topleft, size, chunk_shape, alignment, expected",
    [
        # Grid is aligned to the box topleft, ordered x-major, z-minor.
        (
            (0, 0, 0),
            (20, 20, 20),
            (16, 16, 16),
            None,
            [(x, y, z) for x in (0, 16) for y in (0, 16) for z in (0, 16)],
        ),
        # Unaligned topleft: starts are offset by the topleft, the last chunk of each
        # axis sticks out of the box (starts are not clipped).
        (
            (5, 7, 9),
            (20, 20, 20),
            (16, 16, 16),
            None,
            [(x, y, z) for x in (5, 21) for y in (7, 23) for z in (9, 25)],
        ),
        # Chunk shape larger than the box -> exactly one chunk at the box topleft.
        ((7, 3, 1), (5, 5, 5), (10, 10, 10), None, [(7, 3, 1)]),
        # Negative coordinates.
        (
            (-33, -1, -64),
            (40, 40, 40),
            (32, 32, 32),
            None,
            [(x, y, z) for x in (-33, -1) for y in (-1, 31) for z in (-64, -32)],
        ),
        # With alignment the first start is moved back to the previous multiple of the
        # alignment (topleft 10 -> 0 for alignment 16), so all borders between chunks
        # are multiples of the alignment.
        (
            (10, 10, 10),
            (60, 60, 60),
            (32, 32, 32),
            (16, 16, 16),
            [(x, y, z) for x in (0, 32, 64) for y in (0, 32, 64) for z in (0, 32, 64)],
        ),
        # Per-axis alignment with negative coordinates: -5 % 8 == 3 -> -8,
        # 3 % 4 == 3 -> 0, -7 % 2 == 1 -> -8.
        (
            (-5, 3, -7),
            (20, 10, 10),
            (16, 8, 4),
            (8, 4, 2),
            [(x, y, z) for x in (-8, 8) for y in (0, 8) for z in (-8, -4, 0)],
        ),
    ],
)
def test_iter_chunk_toplefts(
    topleft: tuple[int, int, int],
    size: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    alignment: tuple[int, int, int] | None,
    expected: list[tuple[int, int, int]],
) -> None:
    bbox = BoundingBox(topleft, size)
    starts = list(bbox.iter_chunk_toplefts(chunk_shape, alignment))

    assert starts == expected
    # Yielded values are plain python int tuples (no numpy scalars).
    assert all(type(v) is int for start in starts for v in start)
    if alignment is None:
        # Without chunk_border_alignments, `chunk()` shares the exact same grid.
        assert starts == [tuple(c.topleft) for c in bbox.chunk(chunk_shape, alignment)]


def test_iter_chunk_toplefts_alignment_divisibility_assertion() -> None:
    bbox = BoundingBox((0, 0, 0), (10, 10, 10))
    with pytest.raises(AssertionError):
        list(bbox.iter_chunk_toplefts((32, 32, 32), (7, 7, 7)))


def test_iter_chunk_toplefts_clip_to() -> None:
    # Without chunk_border_alignments, clip_to just restricts which of the box's own
    # (topleft-aligned) chunk starts are yielded.
    bbox = BoundingBox((0, 0, 0), (100, 100, 100))
    clip = BoundingBox((0, 0, 0), (40, 40, 40))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), clip_to=clip)) == [
        (x, y, z) for x in (0, 32) for y in (0, 32) for z in (0, 32)
    ]


def test_iter_chunk_toplefts_clip_to_outside_is_empty() -> None:
    bbox = BoundingBox((200, 200, 200), (10, 10, 10))
    clip = BoundingBox((0, 0, 0), (40, 40, 40))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), clip_to=clip)) == []


def test_iter_chunk_toplefts_as_origin_aligned_grid_cells() -> None:
    # Passing chunk_border_alignments=chunk_shape makes the grid aligned to the
    # coordinate origin instead of to the box's own topleft, which is useful to
    # iterate the grid cells of an origin-aligned partitioning that overlap this box.

    # Box [10, 50) on every axis overlaps origin-aligned 32-grid cells 0 and 32.
    bbox = BoundingBox((10, 10, 10), (40, 40, 40))
    cells = list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32)))
    assert cells == [(x, y, z) for x in (0, 32) for y in (0, 32) for z in (0, 32)]

    # Box exactly spanning one grid cell [32, 64) only overlaps that single cell;
    # neighbouring cells that merely touch its borders are excluded (half-open).
    bbox = BoundingBox((32, 32, 32), (32, 32, 32))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32))) == [(32, 32, 32)]

    # Box [1, 33) crosses the grid line at 32, so it overlaps cells 0 and 32.
    bbox = BoundingBox((1, 1, 1), (32, 32, 32))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32))) == [
        (x, y, z) for x in (0, 32) for y in (0, 32) for z in (0, 32)
    ]

    # Box [-33, -1) crosses the grid lines at -32 and 0, so it overlaps the cells
    # starting at -64 and -32 (the grid is aligned to the origin, not to the box).
    bbox = BoundingBox((-33, -33, -33), (32, 32, 32))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32))) == [
        (x, y, z) for x in (-64, -32) for y in (-64, -32) for z in (-64, -32)
    ]


def test_iter_chunk_toplefts_as_origin_aligned_grid_cells_with_clip_to() -> None:
    bbox = BoundingBox((10, 10, 10), (100, 100, 100))
    clip = BoundingBox((0, 0, 0), (40, 40, 40))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32), clip_to=clip)) == [
        (x, y, z) for x in (0, 32) for y in (0, 32) for z in (0, 32)
    ]

    bbox = BoundingBox((200, 200, 200), (10, 10, 10))
    assert list(bbox.iter_chunk_toplefts((32, 32, 32), (32, 32, 32), clip_to=clip)) == []

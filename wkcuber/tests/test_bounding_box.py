from wkcuber.api.bounding_box import BoundingBox, Mag
import pytest


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

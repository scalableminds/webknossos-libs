import numpy as np

from webknossos.dataset._image_conversion.image_source import ValueRange


def test_of() -> None:
    assert ValueRange.of(np.array([3, 1, 2], dtype="uint8")) == (1.0, 3.0)
    assert ValueRange.of(np.array([-2, 5], dtype="int16")) == (-2.0, 5.0)
    assert ValueRange.of(np.array([], dtype="uint8")) is None


def test_of_ignores_non_finite() -> None:
    # An infinite bound is no use for display, and would be written to
    # datasource-properties.json as invalid JSON.
    assert ValueRange.of(np.array([np.nan, -1.5, 3.5], dtype="float32")) == (-1.5, 3.5)
    assert ValueRange.of(np.array([np.inf, -np.inf, 2.0], dtype="float32")) == (
        2.0,
        2.0,
    )
    assert ValueRange.of(np.array([np.nan, np.nan], dtype="float32")) is None
    assert ValueRange.of(np.array([np.inf, -np.inf], dtype="float32")) is None


def test_combined() -> None:
    assert ValueRange.combined([]) is None
    assert ValueRange.combined([None, None]) is None
    assert ValueRange.combined([ValueRange(1.0, 2.0), None]) == (1.0, 2.0)
    assert ValueRange.combined(
        [ValueRange(1.0, 4.0), ValueRange(0.0, 2.0), ValueRange(3.0, 3.0)]
    ) == (0.0, 4.0)

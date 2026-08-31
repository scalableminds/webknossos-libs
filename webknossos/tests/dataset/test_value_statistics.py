import numpy as np
import pytest

from webknossos.dataset._image_conversion.value_statistics import (
    HISTOGRAM_BINS,
    ValueStatistics,
)


def _reference_clip(
    data: np.ndarray, threshold_ratio: float = 0.0001
) -> tuple[float, float]:
    """The clipping WEBKNOSSOS itself does, on an exact histogram.

    Ported from `clip_histogram_saga.ts` to check the binned approximation
    against.
    """
    values = data.ravel()
    values = values[(values != 0) & np.isfinite(values)]
    keys, counts = np.unique(values, return_counts=True)
    accumulated = np.cumsum(counts)
    area = accumulated[-1]
    threshold = threshold_ratio * area / 2.0
    lower = keys[np.flatnonzero(accumulated >= threshold)[0]]
    upper = keys[np.flatnonzero(accumulated < area - threshold)[-1]]
    return (float(lower), float(upper))


def test_value_range() -> None:
    statistics = ValueStatistics.of(np.array([3, 1, 2], dtype="uint8"))
    assert statistics is not None
    assert statistics.value_range == (1.0, 3.0)
    assert ValueStatistics.of(np.array([], dtype="uint8")) is None


def test_value_range_ignores_nan() -> None:
    statistics = ValueStatistics.of(np.array([np.nan, -1.5, 3.5], dtype="float32"))
    assert statistics is not None
    assert statistics.value_range == (-1.5, 3.5)
    assert ValueStatistics.of(np.array([np.nan, np.nan], dtype="float32")) is None


def test_histogram_skips_zeros() -> None:
    statistics = ValueStatistics.of(np.array([0, 0, 0, 5, 5, 5, 5], dtype="uint8"))
    assert statistics is not None
    assert statistics.counts.sum() == 4

    all_zero = ValueStatistics.of(np.zeros(16, dtype="uint8"))
    assert all_zero is not None
    # The range is still reported, there is just nothing to clip.
    assert all_zero.value_range == (0.0, 0.0)
    assert all_zero.counts.sum() == 0
    assert all_zero.clipped_range(integral=True) is None


def test_histogram_skips_non_finite() -> None:
    statistics = ValueStatistics.of(
        np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype="float32")
    )
    assert statistics is not None
    assert statistics.value_range == (-np.inf, np.inf)
    assert statistics.counts.sum() == 2
    assert (statistics.low, statistics.high) == (1.0, 2.0)


def test_combined_covers_every_part() -> None:
    parts = [
        np.full(100, 10, dtype="uint16"),
        np.zeros(100, dtype="uint16"),
        np.full(100, 500, dtype="uint16"),
    ]
    combined = ValueStatistics.combined(ValueStatistics.of(part) for part in parts)
    assert combined is not None
    assert combined.value_range == (0.0, 500.0)
    # The all-zero part contributes its range but no bins, so the histogram
    # still spans only the values worth showing.
    assert (combined.low, combined.high) == (10.0, 500.0)
    assert combined.counts.sum() == 200
    assert len(combined.counts) == HISTOGRAM_BINS
    assert ValueStatistics.combined([]) is None
    assert ValueStatistics.combined([None, None]) is None


@pytest.mark.parametrize(
    "name",
    ["gaussian_uint8", "gaussian_uint16", "sparse_outliers", "float32", "mostly_zeros"],
)
def test_clipped_range_approximates_webknossos(name: str) -> None:
    rng = np.random.default_rng(0)
    if name == "gaussian_uint8":
        data = np.clip(rng.normal(128, 20, 500_000), 1, 255).astype("uint8")
    elif name == "gaussian_uint16":
        data = np.clip(rng.normal(3000, 400, 500_000), 1, 65535).astype("uint16")
    elif name == "sparse_outliers":
        # A handful of maxed-out voxels is exactly what clipping is for.
        data = np.concatenate(
            [
                np.clip(rng.normal(3000, 400, 500_000), 1, 65535).astype("uint16"),
                np.full(30, 65535, dtype="uint16"),
            ]
        )
    elif name == "float32":
        data = rng.normal(0.5, 0.1, 500_000).astype("float32")
    else:
        data = np.where(
            rng.random(500_000) < 0.9, 0, rng.integers(100, 200, 500_000)
        ).astype("uint16")

    # Chunked the way a conversion would be, so the re-binning is exercised.
    statistics = ValueStatistics.combined(
        ValueStatistics.of(chunk) for chunk in np.array_split(data, 7)
    )
    assert statistics is not None
    assert statistics.value_range == (float(data.min()), float(data.max()))
    integral = np.issubdtype(data.dtype, np.integer)
    clipped = statistics.clipped_range(integral=integral)
    assert clipped is not None

    expected_low, expected_high = _reference_clip(data)
    # The bins make this an approximation; one bin of the combined range is
    # the accuracy it can offer.
    tolerance = (statistics.high - statistics.low) / HISTOGRAM_BINS + (
        1.0 if integral else 0.0
    )
    assert clipped[0] == pytest.approx(expected_low, abs=tolerance)
    assert clipped[1] == pytest.approx(expected_high, abs=tolerance)
    # Clipping only ever narrows the range.
    assert clipped[0] >= statistics.low
    assert clipped[1] <= statistics.high


def test_clipped_range_of_constant_data() -> None:
    statistics = ValueStatistics.of(np.full(100, 7, dtype="uint8"))
    assert statistics is not None
    assert statistics.clipped_range(integral=True) == (7.0, 7.0)

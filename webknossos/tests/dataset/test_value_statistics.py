import numpy as np
import pytest

from webknossos.dataset._image_conversion import value_statistics
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


def test_non_finite_values_are_left_out() -> None:
    statistics = ValueStatistics.of(
        np.array([np.nan, np.inf, -np.inf, 1.0, 2.0], dtype="float32")
    )
    assert statistics is not None
    assert statistics.counts.sum() == 2
    assert (statistics.low, statistics.high) == (1.0, 2.0)
    # An infinity is no use as a display bound, and would not survive being
    # written to datasource-properties.json either.
    assert statistics.value_range == (1.0, 2.0)
    assert ValueStatistics.of(np.array([np.inf, -np.inf], dtype="float32")) is None


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


def test_combined_counts_are_wide_whatever_it_combined() -> None:
    # Chunk counts are narrow to keep them cheap to hand back, but summing
    # them must not depend on how many chunks there were.
    one = ValueStatistics.of(np.full(100, 10, dtype="uint16"))
    assert one is not None
    assert one.counts.dtype == np.uint32
    for entries in ([one], [one, one]):
        combined = ValueStatistics.combined(entries)
        assert combined is not None
        assert combined.counts.dtype == np.int64


@pytest.mark.parametrize(
    "name",
    ["gaussian_uint8", "gaussian_uint16", "sparse_outliers", "float32", "mostly_zeros"],
)
def test_clipped_range_approximates_webknossos(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _distribution(name)
    integral = np.issubdtype(data.dtype, np.integer)
    expected_low, expected_high = _reference_clip(data)
    span = expected_high - expected_low

    def clip_of_chunks() -> tuple[tuple[float, float], float]:
        # Chunked the way a conversion would be, so the re-binning is
        # exercised.
        statistics = ValueStatistics.combined(
            ValueStatistics.of(chunk) for chunk in np.array_split(data, 7)
        )
        assert statistics is not None
        assert statistics.value_range == (float(data.min()), float(data.max()))
        clipped = statistics.clipped_range(integral=integral)
        assert clipped is not None
        # Clipping only ever narrows the range.
        assert clipped[0] >= statistics.low
        assert clipped[1] <= statistics.high
        return clipped, (statistics.high - statistics.low) / HISTOGRAM_BINS

    # A sampled quantile wobbles, so this is the accuracy of the whole thing.
    sampled, _ = clip_of_chunks()
    assert sampled[0] == pytest.approx(expected_low, abs=0.05 * span)
    assert sampled[1] == pytest.approx(expected_high, abs=0.05 * span)

    # Without sampling, only the bin width stands between this and the exact
    # answer WEBKNOSSOS computes.
    monkeypatch.setattr(value_statistics, "HISTOGRAM_SAMPLE_SIZE", data.size)
    monkeypatch.setattr(value_statistics, "HISTOGRAM_SAMPLE_DIVISOR", 1)
    complete, bin_width = clip_of_chunks()
    tolerance = bin_width + (1.0 if integral else 0.0)
    assert complete[0] == pytest.approx(expected_low, abs=tolerance)
    assert complete[1] == pytest.approx(expected_high, abs=tolerance)


def _distribution(name: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if name == "gaussian_uint8":
        return np.clip(rng.normal(128, 20, 500_000), 1, 255).astype("uint8")
    if name == "gaussian_uint16":
        return np.clip(rng.normal(3000, 400, 500_000), 1, 65535).astype("uint16")
    if name == "sparse_outliers":
        # A handful of maxed-out voxels is exactly what clipping is for.
        return np.concatenate(
            [
                np.clip(rng.normal(3000, 400, 500_000), 1, 65535).astype("uint16"),
                np.full(30, 65535, dtype="uint16"),
            ]
        )
    if name == "float32":
        return rng.normal(0.5, 0.1, 500_000).astype("float32")
    return np.where(
        rng.random(500_000) < 0.9, 0, rng.integers(100, 200, 500_000)
    ).astype("uint16")


def test_large_input_is_sampled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(value_statistics, "HISTOGRAM_SAMPLE_SIZE", 50_000)
    rng = np.random.default_rng(0)
    data = np.clip(rng.normal(3000, 400, (20, 100, 250)), 1, 65535).astype("uint16")

    statistics = ValueStatistics.of(data)
    assert statistics is not None
    assert 0 < statistics.counts.sum() < data.size
    # Only the histogram is sampled; the range still covers every value.
    assert statistics.value_range == (float(data.min()), float(data.max()))
    # Counts travel back from every chunk, so they stay narrow.
    assert statistics.counts.dtype == np.uint32

    monkeypatch.setattr(value_statistics, "HISTOGRAM_SAMPLE_SIZE", data.size)
    monkeypatch.setattr(value_statistics, "HISTOGRAM_SAMPLE_DIVISOR", 1)
    complete = ValueStatistics.of(data)
    assert complete is not None
    assert complete.counts.sum() == data.size

    sampled_clip = statistics.clipped_range(integral=True)
    complete_clip = complete.clipped_range(integral=True)
    assert sampled_clip is not None and complete_clip is not None
    # A sampled quantile wobbles, but only by a fraction of the range it is
    # estimating -- and this samples far harder than a conversion does.
    tolerance = 0.05 * (complete_clip[1] - complete_clip[0])
    assert sampled_clip[0] == pytest.approx(complete_clip[0], abs=tolerance)
    assert sampled_clip[1] == pytest.approx(complete_clip[1], abs=tolerance)


def test_clipped_range_of_constant_data() -> None:
    statistics = ValueStatistics.of(np.full(100, 7, dtype="uint8"))
    assert statistics is not None
    assert statistics.clipped_range(integral=True) == (7.0, 7.0)

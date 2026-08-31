"""Value statistics collected while converting images.

Reading the images is the expensive part of a conversion, so the range and the
distribution of the values are gathered as the data passes through, and turned
into the layer's default view configuration afterwards.

The histogram is approximate: every chunk contributes a fixed number of bins
over its own value range, and those are re-binned onto a common range when
they are combined. The clipping itself mirrors WEBKNOSSOS's own
`clip_histogram_saga`, so a converted layer opens with roughly the range that
the "clip histogram" button would produce.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

HISTOGRAM_BINS = 4096
"""Bins per chunk histogram. Fine enough for a display default, small enough
to hand back from every chunk of a large conversion."""

DEFAULT_THRESHOLD_RATIO = 0.0001
"""Share of the values clipped off, split between both ends. Same default as
WEBKNOSSOS's own histogram clipping."""


def value_range_of(array: np.ndarray) -> tuple[float, float] | None:
    """The `(min, max)` of `array` as floats.

    NaNs are ignored, so a single NaN does not wipe out the range of an
    otherwise ordinary image. None when the array is empty or all-NaN.
    """
    if array.size == 0:
        return None
    if np.issubdtype(array.dtype, np.floating):
        with warnings.catch_warnings():
            # An all-NaN array warns and yields NaN, which is handled below.
            warnings.simplefilter("ignore", RuntimeWarning)
            low, high = np.nanmin(array), np.nanmax(array)
        if np.isnan(low):
            return None
        return (float(low), float(high))
    return (float(array.min()), float(array.max()))


def merge_value_ranges(
    left: tuple[float, float] | None, right: tuple[float, float] | None
) -> tuple[float, float] | None:
    """The range covering both `left` and `right`, ignoring missing ones."""
    if left is None:
        return right
    if right is None:
        return left
    return (min(left[0], right[0]), max(left[1], right[1]))


class ValueHistogram(NamedTuple):
    """How values are distributed, in `HISTOGRAM_BINS` equally wide bins
    spanning `low` to `high` inclusive."""

    counts: np.ndarray
    low: float
    high: float

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.low, self.high, HISTOGRAM_BINS + 1)


def histogram_of(array: np.ndarray) -> ValueHistogram | None:
    """The distribution of `array`'s values, or None when there is nothing to
    describe.

    Zeros are left out, as they are background rather than signal in almost
    every image; NaNs and infinities are left out because they cannot be
    binned. Data that is entirely zero therefore has no histogram.
    """
    value_range = value_range_of(array)
    if value_range is None:
        return None
    low, high = value_range
    if not (np.isfinite(low) and np.isfinite(high)):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return None
        low, high = float(finite.min()), float(finite.max())
    counts, edges = np.histogram(array, bins=HISTOGRAM_BINS, range=(low, high))
    zeros = array.size - np.count_nonzero(array)
    if zeros > 0 and low <= 0.0 <= high:
        zero_bin = min(
            int(np.searchsorted(edges, 0.0, side="right")) - 1, len(counts) - 1
        )
        counts[zero_bin] = max(int(counts[zero_bin]) - zeros, 0)
    if counts.sum() <= 0:
        return None
    return ValueHistogram(counts.astype(np.uint32), low, high)


def combine_histograms(
    histograms: Iterable[ValueHistogram | None],
) -> ValueHistogram | None:
    """One histogram covering all of `histograms`.

    Every input is re-binned onto the combined range exactly once, so the
    approximation does not compound with the number of inputs.
    """
    present = [histogram for histogram in histograms if histogram is not None]
    if len(present) == 0:
        return None
    if len(present) == 1:
        return present[0]
    low = min(histogram.low for histogram in present)
    high = max(histogram.high for histogram in present)
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    for histogram in present:
        edges = histogram.edges
        centers = (edges[:-1] + edges[1:]) / 2
        rebinned, _ = np.histogram(
            centers, bins=HISTOGRAM_BINS, range=(low, high), weights=histogram.counts
        )
        counts += rebinned.astype(np.int64)
    return ValueHistogram(counts, low, high)


def clip_histogram(
    histogram: ValueHistogram,
    *,
    integral: bool,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
) -> tuple[float, float] | None:
    """The value range left once the outermost `threshold_ratio` of the values
    is clipped off, split between both ends.

    This is what WEBKNOSSOS's "clip histogram" does, so that a converted layer
    opens with a comparable intensity range. Bin edges are taken as the bounds,
    keeping the range on the generous side. None when the histogram is empty.
    """
    counts = histogram.counts.astype(np.int64)
    area = int(counts.sum())
    if area <= 0:
        return None
    threshold = threshold_ratio * area / 2.0
    cumulative = np.cumsum(counts)
    # Only bins holding data are candidates, so that an empty stretch above a
    # sparse outlier does not push the upper bound back up to it.
    occupied = counts > 0
    lower_bins = np.flatnonzero(occupied & (cumulative >= threshold))
    upper_bins = np.flatnonzero(occupied & (cumulative < area - threshold))
    if len(lower_bins) == 0:
        return None
    edges = histogram.edges
    low = float(edges[lower_bins[0]])
    # The upper bound is the top of its bin, the lower one the bottom of its.
    high = (
        float(edges[int(upper_bins[-1]) + 1]) if len(upper_bins) > 0 else histogram.high
    )
    if integral:
        # A bin holds no integer outside these, so this is the tightest
        # rounding that still covers the bin's values.
        inner_low, inner_high = float(np.ceil(low)), float(np.floor(high))
        if inner_low <= inner_high:
            low, high = inner_low, inner_high
        else:
            low, high = float(np.floor(low)), float(np.ceil(high))
    # The bins reach past the data, so keep the result within it.
    low = min(max(low, histogram.low), histogram.high)
    high = min(max(high, low), histogram.high)
    return (low, high)

"""Value statistics collected while converting images.

Reading the images is the expensive part of a conversion, so the range and the
distribution of the values are gathered as the data passes through, and turned
into the layer's default view configuration afterwards.

The histogram is approximate: every chunk contributes a fixed number of bins
over its own value range, and those are re-binned onto a common range when
they are combined. The clipping mirrors WEBKNOSSOS's own
`clip_histogram_saga`, so a converted layer opens with roughly the range that
the "clip histogram" button would produce.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

HISTOGRAM_BINS = 4096
"""Bins per histogram. Fine enough for a display default, small enough to hand
back from every chunk of a large conversion."""

DEFAULT_THRESHOLD_RATIO = 0.0001
"""Share of the values clipped off, split between both ends. Same default as
WEBKNOSSOS's own histogram clipping."""


class ValueStatistics(NamedTuple):
    """What the values of some data look like: their range, and how they are
    distributed within it."""

    value_range: tuple[float, float]
    """The smallest and largest value, for the layer's `min`/`max`."""

    counts: np.ndarray
    """`HISTOGRAM_BINS` counts over equally wide bins spanning `low` to `high`
    inclusive. All zero when there was nothing to bin."""

    low: float
    high: float
    """The histogram's range. Narrower than `value_range` when the data holds
    infinities, which cannot be binned."""

    @classmethod
    def of(cls, array: np.ndarray) -> ValueStatistics | None:
        """The statistics of `array`, or None when it holds no usable value.

        Zeros are left out of the histogram, as they are background rather than
        signal in almost every image; NaNs and infinities are left out because
        they cannot be binned. They all still count towards the value range,
        except for NaNs, which would otherwise wipe out the range of an
        otherwise ordinary image.
        """
        value_range = _value_range_of(array)
        if value_range is None:
            return None
        low, high = value_range
        if not (np.isfinite(low) and np.isfinite(high)):
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                return cls._unbinned(value_range)
            low, high = float(finite.min()), float(finite.max())
        counts, edges = np.histogram(array, bins=HISTOGRAM_BINS, range=(low, high))
        zeros = array.size - np.count_nonzero(array)
        if zeros > 0 and low <= 0.0 <= high:
            zero_bin = min(
                int(np.searchsorted(edges, 0.0, side="right")) - 1, len(counts) - 1
            )
            counts[zero_bin] = max(int(counts[zero_bin]) - zeros, 0)
        return cls(value_range, counts, low, high)

    @classmethod
    def combined(
        cls, statistics: Iterable[ValueStatistics | None]
    ) -> ValueStatistics | None:
        """The statistics of everything `statistics` was gathered from.

        Every histogram is re-binned onto the combined range exactly once, so
        the approximation does not compound with the number of inputs.
        """
        present = [entry for entry in statistics if entry is not None]
        if len(present) == 0:
            return None
        if len(present) == 1:
            return present[0]
        value_range = (
            min(entry.value_range[0] for entry in present),
            max(entry.value_range[1] for entry in present),
        )
        binned = [entry for entry in present if entry.counts.sum() > 0]
        if len(binned) == 0:
            return cls._unbinned(value_range)
        low = min(entry.low for entry in binned)
        high = max(entry.high for entry in binned)
        counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        for entry in binned:
            edges = entry.edges
            centers = (edges[:-1] + edges[1:]) / 2
            rebinned, _ = np.histogram(
                centers, bins=HISTOGRAM_BINS, range=(low, high), weights=entry.counts
            )
            counts += rebinned.astype(np.int64)
        return cls(value_range, counts, low, high)

    @classmethod
    def _unbinned(cls, value_range: tuple[float, float]) -> ValueStatistics:
        """Statistics for data with a range but nothing to bin, e.g. data that
        is entirely zero."""
        return cls(value_range, np.zeros(HISTOGRAM_BINS, dtype=np.int64), 0.0, 0.0)

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.low, self.high, HISTOGRAM_BINS + 1)

    def clipped_range(
        self,
        *,
        integral: bool,
        threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    ) -> tuple[float, float] | None:
        """The value range left once the outermost `threshold_ratio` of the
        values is clipped off, split between both ends.

        This is what WEBKNOSSOS's "clip histogram" does, so that a converted
        layer opens with a comparable intensity range. None when there is no
        histogram to clip, e.g. for data that is entirely zero.
        """
        counts = self.counts.astype(np.int64)
        area = int(counts.sum())
        if area <= 0:
            return None
        threshold = threshold_ratio * area / 2.0
        cumulative = np.cumsum(counts)
        # Only bins holding data are candidates, so that an empty stretch above
        # a sparse outlier does not push the upper bound back up to it.
        occupied = counts > 0
        lower_bins = np.flatnonzero(occupied & (cumulative >= threshold))
        upper_bins = np.flatnonzero(occupied & (cumulative < area - threshold))
        if len(lower_bins) == 0:
            return None
        edges = self.edges
        # The upper bound is the top of its bin, the lower one the bottom of its.
        low = float(edges[lower_bins[0]])
        high = (
            float(edges[int(upper_bins[-1]) + 1]) if len(upper_bins) > 0 else self.high
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
        low = min(max(low, self.low), self.high)
        high = min(max(high, low), self.high)
        return (low, high)


def _value_range_of(array: np.ndarray) -> tuple[float, float] | None:
    """The `(min, max)` of `array` as floats, ignoring NaNs. None when the
    array is empty or all-NaN."""
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

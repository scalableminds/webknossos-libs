"""Interface for converting a set of input images into a layer.

A source is one set of input images plus the `ReadOptions` chosen for them. It
reports what the data will look like once written — `dtype`, `num_channels`,
`expected_bbox` — and copies it into a `MagView` one chunk at a time.

Implementations differ in whether they know their x/y extent before reading:
some determine it exactly from metadata and write 3D shard-aligned chunks;
others read slice by slice with an oversized placeholder bbox, chunking along
z only and correcting the final size from what was actually written.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import DTypeLike

from ...dataset_properties import LayerViewConfiguration
from ...geometry.constants import C_AXIS, CXYZ_AXES, T_AXIS, XYZ_AXES
from ...geometry.mag import Mag
from ...geometry.normalized_bounding_box import NormalizedBoundingBox
from ...geometry.vec_int import VecInt
from ..layer.view import MagView


@dataclass(frozen=True)
class ReadOptions:
    """How to read an `ImageSource`, bundled as one value."""

    channel: int | None = None
    """The channel to convert, or None for all."""

    swap_xy: bool = False
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    """Axis swap/mirror options applied while copying a chunk."""

    format_options: Mapping[str, int | None] = field(default_factory=dict)
    """Conversion options only some formats understand, e.g. `czi_channel`. A source reads
    the names it knows and ignores the rest."""

    def format_option(self, name: str) -> int | None:
        return self.format_options.get(name)

    def with_layer_selection(self, selection: Mapping[str, int]) -> ReadOptions:
        """The same options with one layer-splitting combination applied.
        `channel` is a standard option; every other key is format-specific."""
        return replace(
            self,
            channel=selection.get("channel", self.channel),
            format_options={
                **self.format_options,
                **{k: v for k, v in selection.items() if k != "channel"},
            },
        )


class ChannelSelection(NamedTuple):
    """How a source's raw channels map onto what actually gets written."""

    num_channels: int
    """How many channels will actually be written."""

    selected_channel: int | None
    """A single pinned channel index, or None if all of them are used."""

    first_n_channels: int | None
    """Truncation to the first n channels (e.g. RGB out of RGBA),
    or None when none applies."""

    possible_channels: list[int] | None
    """Channel indices that could each become their own layer, or None."""


def compute_channel_selection(
    raw_num_channels: int, channel: int | None
) -> ChannelSelection:
    """The channel-selection rule, shared by every `ImageSource`."""
    if channel is not None:
        assert channel < raw_num_channels, (
            f"Selected channel {channel} (0-indexed), but only {raw_num_channels} channels are available."
        )
        return ChannelSelection(1, channel, None, None)
    if raw_num_channels == 2:
        return ChannelSelection(1, 0, None, [0, 1])
    if raw_num_channels >= 3:
        return ChannelSelection(3, None, 3, list(range(raw_num_channels)))
    return ChannelSelection(raw_num_channels, None, None, None)


def with_canonical_axes(
    box: NormalizedBoundingBox, num_channels: int
) -> NormalizedBoundingBox:
    """Builds the `NormalizedBoundingBox` a converted layer gets, in the
    canonical axis order `([...extras], [t], c, x, y, z)`. Any of "c", "x", "y", "z"
    the source does not carry is added as a singleton axis.

    Unlike `NDBoundingBox.normalize_axes()` (which keeps a missing "c" axis
    implicit, matching xyz-only boxes elsewhere, e.g. remote datasets),
    image-conversion's own internal boxes always carry one.
    """
    bounds = {
        axis: (topleft, size)
        for axis, topleft, size in zip(box.axes, box.topleft, box.size)
    }
    bounds[C_AXIS] = (bounds.get(C_AXIS, (0, 0))[0], num_channels)
    for axis in XYZ_AXES:
        bounds.setdefault(axis, (0, 1))

    extras = [axis for axis in box.axes if axis not in CXYZ_AXES and axis != T_AXIS]
    axes = tuple(extras) + ((T_AXIS,) if T_AXIS in box.axes else ()) + CXYZ_AXES
    return NormalizedBoundingBox(
        topleft=VecInt([bounds[axis][0] for axis in axes], axes=axes),
        size=VecInt([bounds[axis][1] for axis in axes], axes=axes),
        axes=axes,
        name=box.name,
        is_visible=box.is_visible,
        color=box.color,
    )


class ValueRange(NamedTuple):
    """The smallest and largest value of some data, as the layer's `min`/`max`
    are written from."""

    min: float
    max: float

    @classmethod
    def of(cls, array: np.ndarray) -> ValueRange | None:
        """The range of `array`, or None when it holds no usable value.

        NaNs and infinities are ignored: one of them should not wipe out the
        range of an otherwise ordinary image, and neither is any use as a
        display bound.
        """
        if array.size == 0:
            return None
        if np.issubdtype(array.dtype, np.floating):
            with warnings.catch_warnings():
                # An all-NaN array warns and yields NaN, handled below.
                warnings.simplefilter("ignore", RuntimeWarning)
                low, high = float(np.nanmin(array)), float(np.nanmax(array))
            if not (np.isfinite(low) and np.isfinite(high)):
                finite = array[np.isfinite(array)]
                if finite.size == 0:
                    return None
                low, high = float(finite.min()), float(finite.max())
            return cls(low, high)
        return cls(float(array.min()), float(array.max()))

    @classmethod
    def combined(cls, ranges: Iterable[ValueRange | None]) -> ValueRange | None:
        """The range covering all of `ranges`, ignoring the missing ones."""
        present = [value_range for value_range in ranges if value_range is not None]
        if len(present) == 0:
            return None
        return cls(
            min(value_range.min for value_range in present),
            max(value_range.max for value_range in present),
        )


class ChunkResult(NamedTuple):
    """What copying one chunk reported back."""

    xy_size: tuple[int, int]
    """The x/y extent actually written — the true size, not the requested
    chunk's."""

    max_value: int | None
    """The largest value seen, for `largest_segment_id` on segmentation layers.
    None when the source cannot report one."""

    value_range: ValueRange | None = None
    """The smallest and largest value written, for the layer's default view
    configuration. Separate from `max_value`, which stays an exact integer for
    segmentations. None when the chunk held no usable value."""


class ImageSource(ABC):
    """One set of input images, ready to be copied into a layer."""

    dtype: DTypeLike
    """The source's own dtype. The caller may still write a different one."""

    num_channels: int
    """How many channels will actually be written — already reduced by a pinned
    `channel` or by RGBA-to-RGB truncation."""

    channels_are_rgb: bool = False
    """Whether the channels are RGB components of one image rather than
    separate acquisitions, which decides whether they share a layer."""

    @property
    @abstractmethod
    def channel(self) -> int | None:
        """The pinned channel, or None if all are written."""

    @abstractmethod
    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        """The ways this source could be split across several layers, e.g.
        `{"channel": [0, 1, 2]}`, or None when there is nothing to split."""

    @property
    def suggested_view_configuration(self) -> LayerViewConfiguration | None:
        """Format-specific default display (color, intensity range, ...) for
        the channel this source writes, when the format's own metadata
        carries one. None for formats with no such metadata."""
        return None

    def layer_split_label(self, key: str, value: int) -> str:
        """A layer-name suffix component for one `get_layer_split_options()`
        split entry."""
        return f"{key}{value}"

    @property
    def expected_bbox(self) -> NormalizedBoundingBox:
        """The bounding box the data is expected to occupy, in Mag(1). Exact,
        or an oversized placeholder.
        """
        return with_canonical_axes(self._raw_expected_bbox, self.num_channels)

    @property
    @abstractmethod
    def _raw_expected_bbox(self) -> NormalizedBoundingBox:
        """The expected extents in the source's own axes, carrying a "c" axis
        when the source intrinsically has channels. `expected_bbox` reorders
        them and fills in whichever of c/x/y/z is missing, so subclasses do
        not have to.
        """

    @abstractmethod
    def copy_chunk_to_view(
        self,
        bbox: NormalizedBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """Read the data for `bbox` (absolute, Mag(1)) and write it into
        `mag_view`; returns a `ChunkResult`.

        Called once per chunk, potentially in parallel across processes, so
        implementations must not carry open file handles across the call.

        Conventions every implementation shares, so that output is
        interchangeable:

        * `ReadOptions.swap_xy` swaps which source axis feeds which output axis.
        * `ReadOptions.flip_x` mirrors the source's **y** axis and `ReadOptions.flip_y` its **x** axis
          — the axes are named for the output, not the source. Each flip
          mirrors the whole extent, never one chunk in isolation.
        * When `dtype` is given, data is converted to it, but the memory layout
          is left alone.
        """

    @abstractmethod
    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NormalizedBoundingBox
    ) -> NormalizedBoundingBox:
        """The bounding box to give the layer before conversion starts. A
        placeholder is oversized so writes are never out of bounds."""

    @abstractmethod
    def chunk_grid(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NormalizedBoundingBox]:
        """The units of work, one per `copy_chunk_to_view` call. `batch_size`
        is honoured only by strategies that chunk along z."""

    @abstractmethod
    def final_bounding_box(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NormalizedBoundingBox:
        """The layer's bounding box once every chunk has been written.
        Unchanged for a source that started out exact."""

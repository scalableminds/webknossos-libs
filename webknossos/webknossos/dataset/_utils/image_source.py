"""The interface `add_layer_from_images` converts through.

A source is one set of input images plus the `ReadOptions` chosen for them. It
reports what the data will look like once written — `dtype`, `num_channels`,
`expected_bbox` — and copies it into a `MagView` one chunk at a time.

The two implementations differ in one fact, which everything else follows from:
whether the source knows its x/y extent before reading.

* `ChunkedImageSource` does, from metadata: exact `expected_bbox`, 3D
  shard-aligned chunks, nothing to correct afterwards.
* `SlicedImageSource` does not, reading a `SliceSequence` one 2D slice at a
  time: oversized placeholder bbox, chunks along z only, and a final size taken
  from what the chunks reported.

`initial_layer_bounding_box`, `chunk_grid` and `final_bounding_box` are that
difference, kept here rather than as branches in `image_conversion.py`. The
conventions below are why the two produce interchangeable output, so
implementations should point here rather than at each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from numpy.typing import DTypeLike

from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
from ..layer.view import MagView


@dataclass(frozen=True)
class ReadOptions:
    """How to read a source, as one value — every layer between
    `add_layer_from_images` and a reader's constructor carries all of it."""

    channel: int | None = None
    """The channel to convert, or None for all. What a source makes of it is
    `compute_channel_selection`'s business."""

    swap_xy: bool = False
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False
    """See `ImageSource.copy_chunk_to_view`."""

    format_options: Mapping[str, int | None] = field(default_factory=dict)
    """Knobs only some formats understand, e.g. `czi_channel`. A source reads
    the names it knows and ignores the rest, so callers pass all of them."""

    def format_option(self, name: str) -> int | None:
        return self.format_options.get(name)

    def with_layer_selection(self, selection: Mapping[str, int]) -> ReadOptions:
        """The same options with one combination from `get_possible_layers()`
        applied. `channel` is a standard option; every other key is
        format-specific."""
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
    """Truncation to the first 3 of 3-or-more channels (e.g. RGB out of RGBA),
    or None when none applies."""

    possible_channels: list[int] | None
    """Channel indices that could each become their own layer, or None. See
    `ImageSource.get_possible_layers()`."""


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


class ChunkResult(NamedTuple):
    """What copying one chunk reported back."""

    xy_size: tuple[int, int]
    """The x/y extent actually written — the true size, not the requested
    chunk's, which is how a placeholder `expected_bbox` gets corrected."""

    max_value: int | None
    """The largest value seen, for `largest_segment_id` on segmentation layers.
    None when the source cannot report one."""


class ImageSource(ABC):
    """One set of input images, ready to be copied into a layer.

    Everything `image_conversion.py` touches is declared here, so it can drive
    either implementation without knowing which it holds.
    """

    dtype: DTypeLike
    """The source's own dtype. The caller may still write a different one."""

    num_channels: int
    """How many channels will actually be written — already reduced by a pinned
    `channel` or by RGBA-to-RGB truncation."""

    channels_are_colour: bool = False
    """Whether the channels are colour components of one image rather than
    separate acquisitions, which decides whether they share a layer. Scientific
    formats store acquisitions, so False is the right default; only the
    everyday raster formats say otherwise."""

    @property
    @abstractmethod
    def channel(self) -> int | None:
        """The pinned channel, or None if all are written. May differ from what
        was requested — see `compute_channel_selection`."""

    @abstractmethod
    def get_possible_layers(self) -> dict[str, list[int]] | None:
        """The ways this source could be split across several layers, e.g.
        `{"channel": [0, 1, 2]}`, or None when there is nothing to split.

        With `allow_multiple_layers=True` the caller re-opens the source once
        per combination via `ReadOptions.with_layer_selection()`.
        """

    @property
    @abstractmethod
    def expected_bbox(self) -> NDBoundingBox:
        """The bounding box the data is expected to occupy, in Mag(1). Exact,
        or an oversized placeholder — see the module docstring."""

    @abstractmethod
    def copy_chunk_to_view(
        self,
        bbox: NDBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """Read the data for `bbox` (absolute, Mag(1)) and write it into
        `mag_view`; returns a `ChunkResult`.

        Called once per chunk, potentially in parallel across processes, so
        implementations must not carry open file handles across the call.

        Conventions every implementation shares, so that output is
        interchangeable:

        * `swap_xy` swaps which source axis feeds which output axis.
        * `flip_x` mirrors the source's **y** axis and `flip_y` its **x** axis
          — the axes are named for the output, not the source. Each flip
          mirrors the whole extent, never one chunk in isolation.
        * When `dtype` is given, data is converted with `order="F"`, which is
          what the downstream writers expect.
        """

    @abstractmethod
    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NDBoundingBox
    ) -> NDBoundingBox:
        """The bounding box to give the layer before conversion starts. A
        placeholder is oversized so writes are never out of bounds."""

    @abstractmethod
    def chunk_grid(
        self,
        layer_bounding_box: NDBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NDBoundingBox]:
        """The units of work, one per `copy_chunk_to_view` call. `batch_size`
        is honoured only by strategies that chunk along z."""

    @abstractmethod
    def final_bounding_box(
        self,
        layer_bounding_box: NDBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NDBoundingBox:
        """The layer's bounding box once every chunk has been written, from the
        `ChunkResult.xy_size` values they reported. Unchanged for a source that
        started out exact."""

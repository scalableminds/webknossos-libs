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

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from numpy.typing import DTypeLike

from ...dataset_properties import LayerViewConfiguration
from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
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


class ChunkResult(NamedTuple):
    """What copying one chunk reported back."""

    xy_size: tuple[int, int]
    """The x/y extent actually written — the true size, not the requested
    chunk's."""

    max_value: int | None
    """The largest value seen, for `largest_segment_id` on segmentation layers.
    None when the source cannot report one."""


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
    def get_possible_layers(self) -> dict[str, list[int]] | None:
        """The ways this source could be split across several layers, e.g.
        `{"channel": [0, 1, 2]}`, or None when there is nothing to split."""

    @property
    def suggested_view_configuration(self) -> LayerViewConfiguration | None:
        """Format-specific default display (color, intensity range, ...) for
        the channel this source writes, when the format's own metadata
        carries one. None for formats with no such metadata."""
        return None

    def layer_split_label(self, key: str, value: int) -> str:
        """A layer-name suffix component for one `get_possible_layers()`
        split entry. Defaults to `f"{key}{value}"`; a format may override
        this with something more descriptive, e.g. an OME-Zarr channel's
        `omero` label."""
        return f"{key}{value}"

    @property
    @abstractmethod
    def expected_bbox(self) -> NDBoundingBox:
        """The bounding box the data is expected to occupy, in Mag(1). Exact,
        or an oversized placeholder."""

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

        * `ReadOptions.swap_xy` swaps which source axis feeds which output axis.
        * `ReadOptions.flip_x` mirrors the source's **y** axis and `ReadOptions.flip_y` its **x** axis
          — the axes are named for the output, not the source. Each flip
          mirrors the whole extent, never one chunk in isolation.
        * When `dtype` is given, data is converted with `order="F"`.
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
        """The layer's bounding box once every chunk has been written.
        Unchanged for a source that started out exact."""

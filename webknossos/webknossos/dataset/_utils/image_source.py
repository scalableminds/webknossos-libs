"""The interface `add_layer_from_images` converts through, shared by both
reading strategies.

A source is one set of input images together with the reading options that
were chosen for it (channel, swap_xy, the flips). It reports what the data
will look like once written — `dtype`, `num_channels`, `expected_bbox` — and
copies it into a `MagView` one chunk at a time.

Two strategies implement this:

* `SlicedImageSource` reads a `SliceSequence` slice by slice. It suits formats
  that only offer sequential 2D access, and it cannot always know its true x/y
  extent up front, so its `expected_bbox` may be a placeholder that the caller
  corrects afterwards from what was actually written.
* `ChunkedImageSource` reads whole shard-sized blocks straight out of a
  volumetric file. It suits formats that state their exact extents in
  metadata, so its `expected_bbox` is always exact.

Everything below is the contract both must honour; the conventions stated here
are the reason the two produce interchangeable output, so implementations
should point at this file rather than at each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import NamedTuple

from numpy.typing import DTypeLike

from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
from ..layer.view import MagView


class ChannelSelection(NamedTuple):
    """How a source's raw channels map onto what actually gets written."""

    num_channels: int
    """The number of channels that will actually be written."""

    selected_channel: int | None
    """A single pinned channel index, or None if all (up to
    `first_n_channels`) channels are used."""

    first_n_channels: int | None
    """Set when the source has 3 or more channels, truncating to the first 3
    (e.g. RGB out of RGBA). None when no truncation applies."""

    possible_channels: list[int] | None
    """Channel indices that could each become their own layer (see
    `ImageSource.get_possible_layers()`), or None if not applicable."""


def compute_channel_selection(
    raw_num_channels: int, channel: int | None
) -> ChannelSelection:
    """The channel-selection rule, shared by every `ImageSource`
    implementation."""
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
    """The x/y extent actually written. This is how a placeholder
    `expected_bbox` gets corrected — it is the true size, not the requested
    chunk's."""

    max_value: int | None
    """The largest value seen, used to set `largest_segment_id` on
    segmentation layers. None when the source cannot report one."""


class ImageSource(ABC):
    """One set of input images, ready to be copied into a layer.

    Subclasses differ only in how they get at the data; everything the
    conversion machinery in `image_conversion.py` touches is declared here, so
    it can drive either without knowing which it holds.
    """

    dtype: DTypeLike
    """The source's own dtype. The caller may still write a different one."""

    num_channels: int
    """How many channels will actually be written — already reduced by a
    pinned `channel` or by RGBA-to-RGB truncation, so not necessarily the
    source's raw channel count."""

    @property
    @abstractmethod
    def channel(self) -> int | None:
        """The pinned channel, or None if all channels are written.

        May differ from the `channel` that was requested: a two-channel source
        pins channel 0 by itself (see `compute_channel_selection`).
        """

    @abstractmethod
    def get_possible_layers(self) -> dict[str, list[int]] | None:
        """The ways this source could be split across several layers, e.g.
        `{"channel": [0, 1, 2]}`, or None when there is nothing to split.

        `add_layer_from_images` re-opens the source once per combination when
        `allow_multiple_layers=True`, passing the value back as the matching
        keyword argument.
        """

    @property
    @abstractmethod
    def expected_bbox(self) -> NDBoundingBox:
        """The bounding box the data is expected to occupy, in Mag(1).

        Exact for sources that know their extents from metadata. Sources that
        only discover the true x/y extent while reading return a deliberately
        oversized placeholder here, which the caller shrinks afterwards using
        the sizes reported by `copy_chunk_to_view`.
        """

    @abstractmethod
    def copy_chunk_to_view(
        self,
        bbox: NDBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """Read the data for `bbox` and write it into `mag_view`.

        Called once per chunk, potentially in parallel across processes, so
        implementations must not carry open file handles across the call
        boundary. `bbox` is absolute and in Mag(1).

        See `ChunkResult` for what comes back.

        Conventions every implementation shares, so that output is
        interchangeable:

        * `swap_xy` swaps which source axis feeds which output axis.
        * `flip_x` mirrors the source's **y** axis and `flip_y` its **x** axis
          — the axes are named for the output, not the source. Each flip
          mirrors the whole extent, never one chunk in isolation.
        * When `dtype` is given, data is converted with `order="F"`, which is
          what the downstream writers expect.
        """

    # ------------------------------------------------------------------
    # How the conversion is driven.
    #
    # These three follow from one fact, which is the real difference between
    # the strategies: whether the source knows its x/y extent before reading.
    # A source that does gets an exact bounding box, can be split into 3D
    # shard-aligned chunks, and needs no correction afterwards. A source that
    # does not gets a placeholder, can only be split along z (there is no known
    # x/y to split), and has to be shrunk once the real extent is known.
    #
    # They live here rather than as branches in image_conversion.py so that the
    # caller never has to ask which kind of source it is holding.
    # ------------------------------------------------------------------

    @abstractmethod
    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NDBoundingBox
    ) -> NDBoundingBox:
        """The bounding box to give the layer before conversion starts.

        Exact when the extent is known; deliberately oversized otherwise, so
        that writes are never rejected as out of bounds. `final_bounding_box`
        cuts it back down.
        """

    @abstractmethod
    def chunk_grid(
        self,
        layer_bounding_box: NDBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NDBoundingBox]:
        """The units of work, one per `copy_chunk_to_view` call.

        `batch_size` is the caller's request, honoured only by strategies that
        chunk along z; others ignore it.
        """

    @abstractmethod
    def final_bounding_box(
        self,
        layer_bounding_box: NDBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NDBoundingBox:
        """The layer's bounding box once every chunk has been written.

        `chunk_sizes` are the `ChunkResult.xy_size` values the chunks reported,
        which is how a placeholder box learns its real x/y extent. Sources that
        started out exact return the box unchanged.
        """

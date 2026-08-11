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

from numpy.typing import DTypeLike

from ...geometry.nd_bounding_box import NDBoundingBox
from ..layer.view import MagView


def compute_channel_selection(
    raw_num_channels: int, channel: int | None
) -> tuple[int, int | None, int | None, list[int] | None]:
    """
    The channel-selection rule, shared by every `ImageSource` implementation.
    Returns (num_channels, selected_channel, first_n_channels,
    possible_channels):
    - num_channels: the number of channels that will actually be written
    - selected_channel: a single pinned channel index, or None if all
      (up to first_n_channels) channels are used
    - first_n_channels: set when raw_num_channels >= 3, truncating to the
      first 3 channels (e.g. RGB out of RGBA)
    - possible_channels: channel indices that could each become their own
      layer (see get_possible_layers()), or None if not applicable
    """
    if channel is not None:
        assert channel < raw_num_channels, (
            f"Selected channel {channel} (0-indexed), but only {raw_num_channels} channels are available."
        )
        return 1, channel, None, None
    if raw_num_channels == 2:
        return 1, 0, None, [0, 1]
    if raw_num_channels >= 3:
        return 3, None, 3, list(range(raw_num_channels))
    return raw_num_channels, None, None, None


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
    ) -> tuple[tuple[int, int], int | None]:
        """Read the data for `bbox` and write it into `mag_view`.

        Called once per chunk, potentially in parallel across processes, so
        implementations must not carry open file handles across the call
        boundary. `bbox` is absolute and in Mag(1).

        Returns `((x_size, y_size), max_value)`:

        * the x/y extent actually written, which is how a placeholder
          `expected_bbox` gets corrected — it is the true size, not `bbox`'s;
        * the largest value seen, used to set `largest_segment_id` on
          segmentation layers. None when the source cannot report one.

        Conventions every implementation shares, so that output is
        interchangeable:

        * `swap_xy` swaps which source axis feeds which output axis.
        * `flip_x` mirrors the source's **y** axis and `flip_y` its **x** axis
          — the axes are named for the output, not the source. Each flip
          mirrors the whole extent, never one chunk in isolation.
        * When `dtype` is given, data is converted with `order="F"`, which is
          what the downstream writers expect.
        """

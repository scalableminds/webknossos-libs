from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.bounding_box import BoundingBox
from ...geometry.constants import T_AXIS, TXYZ_AXES, X_AXIS, Y_AXIS, Z_AXIS
from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
from ...geometry.normalized_bounding_box import NormalizedBoundingBox
from ..layer.view import MagView
from .image_source import (
    ChunkResult,
    ImageSource,
    ReadOptions,
    value_range_of,
    with_explicit_channel_axis,
)


class ChunkedImageSource(ImageSource):
    """
    ImageSource for volumetric formats (ims, mrc, czi) that state their exact
    extents in metadata and can read an arbitrary box of voxels, so each chunk
    reads and writes a whole shard-aligned block with no slice writer between.

    Subclasses implement only `_read_source_box`; the axis bookkeeping lives
    here.
    """

    # Source extents in the file's own axis order, set from metadata by the
    # subclass. Always pre-swap_xy: _x is the source's x, even when swap_xy
    # makes it the output's y.
    _x: int
    _y: int
    _z: int
    # Formats without a time dimension leave these alone and read timepoint 0.
    # A subclass with several timepoints and none pinned sets _include_t_axis,
    # which puts a "t" axis on expected_bbox and gives each chunk its own.
    _t: int = 1
    _include_t_axis: bool = False
    _fixed_timepoint: int | None = 0

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        self.path = path
        self._options = options
        # The requested channel; subclasses replace it with the resolved one.
        self._channel = options.channel

    @classmethod
    @abstractmethod
    def supported_file_extensions(cls) -> set[str]:
        """File extensions (without the dot, lowercase) this class can read."""

    @classmethod
    def probe_directory(cls, _path: UPath) -> bool:
        """Whether this class can open `path` as a directory-based store, used
        when no registered extension matches. Default: no directory sniffing."""
        return False

    @property
    def channel(self) -> int | None:
        """The selected channel, or None if all channels are used."""
        return self._channel

    @abstractmethod
    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        """The ways this source could be split across several layers, or None
        when there is nothing to split."""

    @abstractmethod
    def _read_source_box(
        self,
        *,
        timepoint: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """
        Read exactly the given box in *source* coordinates and return it as
        (c, z, y, x), with a leading channel axis even for single-channel
        formats. The slices are already resolved to the source's axis order and
        mag, and already account for swap_xy and the flips (which arrive as
        mirrored ranges), so implementations only select channels.
        """

    @property
    def expected_bbox(self) -> NormalizedBoundingBox:
        """
        The exact bounding box of the data, in the source's native Mag(1)
        space — never a placeholder, since these formats know their extents.

        """
        x_size, y_size = self._x, self._y
        if self._options.swap_xy:
            x_size, y_size = y_size, x_size

        if not self._include_t_axis:
            return BoundingBox((0, 0, 0), (x_size, y_size, self._z)).normalize_axes(
                self.num_channels
            )

        box = NDBoundingBox.from_axes(TXYZ_AXES, [self._t, x_size, y_size, self._z])
        return with_explicit_channel_axis(box, self.num_channels)

    def copy_chunk_to_view(
        self,
        bbox: NormalizedBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """
        Reads exactly bbox's worth of data and writes it to mag_view. The x/y
        extent returned is always bbox's, since expected_bbox was exact.
        """
        options = self._options
        relative_bbox = bbox.offset(-mag_view.bounding_box.topleft)

        if "t" in relative_bbox.axes:
            timepoint, _ = relative_bbox.get_bounds(T_AXIS)
        else:
            assert self._fixed_timepoint is not None
            timepoint = self._fixed_timepoint

        # bbox is in Mag(1) while the source is indexed in its own mag, so
        # every bound is scaled down below (the identity for Mag(1)).
        mag_vec = mag_view.mag.to_vec3_int()

        # bbox's x/y are the *output* extents, so with swap_xy set bbox.x holds
        # the source's y-extent and vice versa.

        out_x_start, out_x_end = relative_bbox.get_bounds(X_AXIS)
        out_y_start, out_y_end = relative_bbox.get_bounds(Y_AXIS)
        z_start, z_end = relative_bbox.get_bounds(Z_AXIS)
        z_start, z_end = z_start // mag_vec.z, z_end // mag_vec.z
        if options.swap_xy:
            source_y_start, source_y_end = (
                out_x_start // mag_vec.x,
                out_x_end // mag_vec.x,
            )
            source_x_start, source_x_end = (
                out_y_start // mag_vec.y,
                out_y_end // mag_vec.y,
            )
        else:
            source_x_start, source_x_end = (
                out_x_start // mag_vec.x,
                out_x_end // mag_vec.x,
            )
            source_y_start, source_y_end = (
                out_y_start // mag_vec.y,
                out_y_end // mag_vec.y,
            )

        # Each flip reads a mirrored source range and is reversed back into
        # output order below, mirroring the *entire* extent. Reversing a chunk
        # in place would mirror only within that chunk — invisible in a single
        # shard, wrong across several. flip_x mirrors the source's y axis and
        # flip_y its x axis.
        if options.flip_z:
            read_z = slice(self._z - z_end, self._z - z_start)
        else:
            read_z = slice(z_start, z_end)
        if options.flip_x:
            read_y = slice(self._y - source_y_end, self._y - source_y_start)
        else:
            read_y = slice(source_y_start, source_y_end)
        if options.flip_y:
            read_x = slice(self._x - source_x_end, self._x - source_x_start)
        else:
            read_x = slice(source_x_start, source_x_end)

        block = self._read_source_box(timepoint=timepoint, z=read_z, y=read_y, x=read_x)

        # Mirrored read ranges -> correct output order, in source axis order
        # (c, z, y, x).
        if options.flip_z:
            block = block[:, ::-1]
        if options.flip_x:
            block = block[:, :, ::-1]
        if options.flip_y:
            block = block[:, :, :, ::-1]

        # Transpose to (c, x, y, z).
        # swap_xy=False (default): (c, z, y, x) → (c, x, y, z)
        # swap_xy=True:            (c, z, y, x) → (c, y, x, z)
        block = (
            block.transpose(0, 3, 2, 1)
            if not options.swap_xy
            else block.transpose(0, 2, 3, 1)
        )

        if dtype is not None:
            block = block.astype(dtype, order="F")

        max_value = int(block.max())
        value_range = value_range_of(block)
        if "t" in relative_bbox.axes:
            # Add back the size-1 "t" axis this chunk corresponds to.
            block = block[np.newaxis]

        # allow_unaligned=True: border chunks are smaller than shard_shape,
        # since extents rarely divide evenly. Safe because parallel jobs write
        # disjoint regions.
        mag_view.write(block, absolute_bounding_box=bbox, allow_unaligned=True)

        return ChunkResult(
            (out_x_end - out_x_start, out_y_end - out_y_start), max_value, value_range
        )

    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NormalizedBoundingBox
    ) -> NormalizedBoundingBox:
        """Exact from the start — no placeholder to inflate."""
        return mag1_expected_bbox

    def chunk_grid(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NormalizedBoundingBox]:
        """Full 3D shard-aligned chunks, one shard's worth of data per job.
        Safe to chunk the layer's box because it is the exact one."""
        del batch_size
        chunked_shard_shape = mag_view.info.shard_shape * mag.to_vec3_int()
        return list(layer_bounding_box.chunk(chunked_shard_shape, chunked_shard_shape))

    def final_bounding_box(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NormalizedBoundingBox:
        """Unchanged. Correcting from per-chunk sizes would be wrong: a job
        reports its own shard, not the full extent."""
        del chunk_sizes, mag
        return layer_bounding_box

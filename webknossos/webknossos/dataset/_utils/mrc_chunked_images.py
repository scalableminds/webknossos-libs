from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.bounding_box import BoundingBox
from ...geometry.nd_bounding_box import NDBoundingBox
from ...utils import WkImportError, is_remote_path
from ..layer.view import MagView
from .chunked_images import ChunkedImages, register_chunked_images

try:
    import mrcfile
except ImportError as e:
    raise WkImportError("mrcfile", "mrcfile") from e


@register_chunked_images
class MrcChunkedImages(ChunkedImages):
    """
    ChunkedImages implementation for MRC files. MRC data is stored as a
    single contiguous array (no internal chunking, unlike HDF5-based
    formats), so shard-sized blocks are read directly via mrcfile's
    memory-mapped array and written to mag_view directly — no
    slice-by-slice pims reading, no BufferedSliceWriter.

    MRC files have neither channels nor timepoints, so num_channels is
    always 1 and get_possible_layers() always returns None.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return {"mrc", "rec", "st", "map", "ali"}

    def __init__(
        self,
        path: UPath,
        *,
        channel: int | None,
        swap_xy: bool,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        is_segmentation: bool,
    ) -> None:
        super().__init__(
            path,
            channel=channel,
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            is_segmentation=is_segmentation,
        )
        if is_remote_path(path):
            raise ValueError(
                f"Cannot open MRC file from {path}. The path must be a local file path."
            )

        with mrcfile.mmap(str(path), mode="r", permissive=True) as mrc:
            if mrc.data is None:
                raise ValueError(
                    f"Cannot open MRC file {path}. "
                    + "The file is likely corrupted or not a valid MRC file."
                )
            self.dtype: np.dtype = mrc.data.dtype
            ndim = mrc.data.ndim
            if ndim == 3:
                self._z, self._y, self._x = mrc.data.shape
            elif ndim == 2:
                self._z = 1
                self._y, self._x = mrc.data.shape
            else:
                raise ValueError(
                    f"Unsupported MRC data dimensionality: {ndim}. "
                    "Only 2D and 3D MRC files are supported."
                )

        self.num_channels = 1
        self._swap_xy = swap_xy
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._flip_z = flip_z

    @property
    def channel(self) -> int | None:
        return None

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        return None

    @property
    def expected_bbox(self) -> NDBoundingBox:
        x_size, y_size = self._x, self._y
        if self._swap_xy:
            x_size, y_size = y_size, x_size
        return BoundingBox((0, 0, 0), (x_size, y_size, self._z))

    def read_chunk(
        self,
        bbox: NDBoundingBox,
        *,
        mag_view: MagView,
        dtype: DTypeLike | None,
    ) -> tuple[tuple[int, int], int]:
        relative_bbox = bbox.offset(-mag_view.bounding_box.topleft)

        # bbox's x/y axes describe the *output* extents. When swap_xy is set,
        # expected_bbox swaps which source axis feeds which output axis, so
        # bbox.x holds the source y-extent and bbox.y holds the source
        # x-extent — the mmap read below must use the matching source bound.
        out_x_start, out_x_end = relative_bbox.get_bounds("x")
        out_y_start, out_y_end = relative_bbox.get_bounds("y")
        z_start, z_end = relative_bbox.get_bounds("z")
        if self._swap_xy:
            source_y_start, source_y_end = out_x_start, out_x_end
            source_x_start, source_x_end = out_y_start, out_y_end
        else:
            source_x_start, source_x_end = out_x_start, out_x_end
            source_y_start, source_y_end = out_y_start, out_y_end

        # Every flip mirrors the *entire* source extent, not just this chunk in
        # isolation, so each one reads a mirrored source range and reverses it
        # back into output order below. Reversing a chunk in place would only
        # mirror within that chunk, which is invisible while the image fits in
        # a single shard but wrong as soon as it spans several.
        # flip_x/-y follow the PimsImages convention: flip_x mirrors the
        # source's y axis and flip_y mirrors its x axis.
        if self._flip_z:
            read_z_start, read_z_end = self._z - z_end, self._z - z_start
        else:
            read_z_start, read_z_end = z_start, z_end
        if self._flip_x:
            read_y_start, read_y_end = self._y - source_y_end, self._y - source_y_start
        else:
            read_y_start, read_y_end = source_y_start, source_y_end
        if self._flip_y:
            read_x_start, read_x_end = self._x - source_x_end, self._x - source_x_start
        else:
            read_x_start, read_x_end = source_x_start, source_x_end

        with mrcfile.mmap(str(self.path), mode="r", permissive=True) as mrc:
            if mrc.data.ndim == 3:
                block = np.array(
                    mrc.data[
                        read_z_start:read_z_end,
                        read_y_start:read_y_end,
                        read_x_start:read_x_end,
                    ]
                )
            else:
                block = np.array(
                    mrc.data[read_y_start:read_y_end, read_x_start:read_x_end]
                )[np.newaxis]  # add a size-1 z axis to unify with the 3D case below

        # Mirrored read ranges -> correct output order, in source axis order
        # (z, y, x).
        if self._flip_z:
            block = block[::-1]
        if self._flip_x:
            block = block[:, ::-1]
        if self._flip_y:
            block = block[:, :, ::-1]

        # Transpose to mag_view.write() convention (x, y, z).
        # swap_xy=False (default): (z, y, x) → (x, y, z)
        # swap_xy=True:            (z, y, x) → (y, x, z)
        block = (
            block.transpose(2, 1, 0) if not self._swap_xy else block.transpose(1, 2, 0)
        )

        if dtype is not None:
            block = block.astype(dtype, order="F")

        max_value = int(block.max())

        # allow_unaligned=True: real image extents rarely divide evenly into
        # full shards, so border chunks are smaller than shard_shape. Safe
        # here because each parallel job writes a disjoint,
        # shard-aligned-or-smaller region — no two jobs ever target
        # overlapping data.
        mag_view.write(block, absolute_bounding_box=bbox, allow_unaligned=True)

        # Returned as (x_size, y_size) to match the convention established by
        # PimsImages.copy_to_view.
        x_size = out_x_end - out_x_start
        y_size = out_y_end - out_y_start
        return (x_size, y_size), max_value

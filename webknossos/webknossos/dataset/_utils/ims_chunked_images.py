from __future__ import annotations

import contextlib
import gc
import io

import h5py
import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.bounding_box import BoundingBox
from ...geometry.nd_bounding_box import NDBoundingBox
from ...geometry.vec_int import VecInt
from ...utils import WkImportError, is_remote_path
from ..layer.view import MagView
from .chunked_images import ChunkedImages, register_chunked_images
from .pims_images import compute_channel_selection

try:
    from imaris_ims_file_reader.ims import ims as ImsFile
except ImportError as e:
    raise WkImportError("imaris-ims-file-reader", "ims") from e


def _read_ims_metadata_quietly(
    path: str,
) -> tuple[tuple[int, int, int, int, int], np.dtype]:
    # The published imaris-ims-file-reader (as of 0.1.8) unconditionally prints
    # "Opening readonly file: ..." / "Closing file: ..." and has no `verbose`
    # kwarg to suppress it. It also calls close() again from __del__, so the
    # object must be closed, deleted, and garbage-collected while stdout is
    # still redirected — otherwise that second "Closing file" print leaks out
    # once the object is finalized after this function returns.
    with contextlib.redirect_stdout(io.StringIO()):
        ims_obj = ImsFile(path, squeeze_output=False)
        shape = tuple(int(s) for s in ims_obj.shape)
        dtype = np.dtype(ims_obj.dtype)
        ims_obj.close()
        del ims_obj
        gc.collect()
    return shape, dtype  # type: ignore[return-value]


@register_chunked_images
class ImsChunkedImages(ChunkedImages):
    """
    ChunkedImages implementation for Imaris .ims files. Reads shard-sized 3D
    blocks directly from the underlying HDF5 file via h5py and writes them
    to mag_view directly — no slice-by-slice pims reading, no BufferedSliceWriter.
    This is the only supported way .ims files are read for conversion.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return {"ims"}

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
                f"Cannot open IMS file from {path}. The path must be a local file path."
            )

        file_shape, self.dtype = _read_ims_metadata_quietly(str(path))
        t, raw_num_channels, self._z, self._y, self._x = file_shape
        self._t = t

        self._swap_xy = swap_xy
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._flip_z = flip_z

        self.num_channels, self._channel, self._first_n_channels, possible_channels = (
            compute_channel_selection(raw_num_channels, channel)
        )
        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

        # A "t" axis is only added to the bounding box when there actually are
        # multiple timepoints; each chunk along that axis then carries its own
        # timepoint (chunk size 1), read per chunk in read_chunk() below. A
        # single-timepoint file stays 3D and always reads timepoint 0.
        self._include_t_axis = t > 1
        self._fixed_timepoint: int | None = None if self._include_t_axis else 0

    @property
    def channel(self) -> int | None:
        return self._channel

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    @property
    def expected_bbox(self) -> NDBoundingBox:
        x_size, y_size = self._x, self._y
        if self._swap_xy:
            x_size, y_size = y_size, x_size

        if not self._include_t_axis and self.num_channels == 1:
            return BoundingBox((0, 0, 0), (x_size, y_size, self._z))

        # Report every axis the source actually has: "t" when there are
        # multiple timepoints and none was pinned, and "c" when more than one
        # channel is written. "c" is where NormalizedBoundingBox expects to
        # find num_channels, and NDBoundingBox.chunk() keeps that axis whole
        # instead of splitting it, so read_chunk() still receives the full
        # channel extent per chunk.
        axes = ["x", "y", "z"]
        sizes = [x_size, y_size, self._z]
        if self.num_channels > 1:
            axes.insert(0, "c")
            sizes.insert(0, self.num_channels)
        if self._include_t_axis:
            axes.insert(0, "t")
            sizes.insert(0, self._t)
        return NDBoundingBox(
            VecInt.zeros(tuple(axes)),
            VecInt(sizes, axes=axes),
            axes,
            VecInt(list(range(len(axes))), axes=axes),
        )

    def read_chunk(
        self,
        bbox: NDBoundingBox,
        *,
        mag_view: MagView,
        dtype: DTypeLike | None,
    ) -> tuple[tuple[int, int], int]:
        relative_bbox = bbox.offset(-mag_view.bounding_box.topleft)

        if "t" in relative_bbox.axes:
            timepoint, _ = relative_bbox.get_bounds("t")
        else:
            assert self._fixed_timepoint is not None
            timepoint = self._fixed_timepoint

        # bbox's x/y axes describe the *output* extents. When swap_xy is set,
        # expected_bbox swaps which source axis feeds which output axis, so
        # bbox.x holds the source y-extent and bbox.y holds the source
        # x-extent — the HDF5 read below must use the matching source bound.
        out_x_start, out_x_end = relative_bbox.get_bounds("x")
        out_y_start, out_y_end = relative_bbox.get_bounds("y")
        z_start, z_end = relative_bbox.get_bounds("z")
        if self._swap_xy:
            source_y_start, source_y_end = out_x_start, out_x_end
            source_x_start, source_x_end = out_y_start, out_y_end
        else:
            source_x_start, source_x_end = out_x_start, out_x_end
            source_y_start, source_y_end = out_y_start, out_y_end

        # Every flip mirrors the *entire* source extent (matching
        # PimsImages.copy_to_view, which reverses the full image sequence before
        # slicing per-batch), not just this chunk in isolation. Each one reads a
        # mirrored source range and reverses it back into output order below.
        # Reversing a chunk in place would only mirror within that chunk, which
        # is invisible while the image fits in a single shard but wrong as soon
        # as it spans several.
        # flip_x/-y follow the PimsImages convention: flip_x mirrors memory
        # axis -2 (y in HDF5) and flip_y mirrors memory axis -1 (x in HDF5).
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

        if self._channel is not None:
            channels_to_read = [self._channel]
        elif self._first_n_channels is not None:
            channels_to_read = list(range(self._first_n_channels))
        else:
            channels_to_read = list(range(self.num_channels))

        with h5py.File(str(self.path), "r") as hf:
            ds = hf["DataSet"]
            slabs = []
            for ci in channels_to_read:
                loc = f"ResolutionLevel 0/TimePoint {timepoint}/Channel {ci}/Data"
                # Read exactly the 3D block needed — one decompression per HDF5 chunk row
                slabs.append(
                    ds[loc][
                        read_z_start:read_z_end,
                        read_y_start:read_y_end,
                        read_x_start:read_x_end,
                    ]
                )
            block = np.stack(slabs, axis=0)  # (c, z, y, x)

        # Mirrored read ranges -> correct output order, in HDF5 axis order
        # (c, z, y, x).
        if self._flip_z:
            block = block[:, ::-1]
        if self._flip_x:
            block = block[:, :, ::-1]
        if self._flip_y:
            block = block[:, :, :, ::-1]

        # Transpose to mag_view.write() convention (c, x, y, z).
        # swap_xy=False (default): (c, z, y, x) → (c, x, y, z)
        # swap_xy=True:            (c, z, y, x) → (c, y, x, z)
        block = (
            block.transpose(0, 3, 2, 1)
            if not self._swap_xy
            else block.transpose(0, 2, 3, 1)
        )

        if dtype is not None:
            # order="F" matches PimsImages.copy_to_view, which produces
            # Fortran-contiguous slices for the same downstream writers.
            block = block.astype(dtype, order="F")

        max_value = int(block.max())
        if self.num_channels == 1:
            block = block[0]  # (x, y, z) — single-channel layers have no c axis
        if "t" in relative_bbox.axes:
            # View.write() requires data.shape to have exactly as many
            # dimensions as the layer's bbox axes; add the (size-1) "t"
            # dimension this chunk corresponds to.
            block = block[np.newaxis]

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

from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import DTypeLike
from pims import FramesSequenceND
from upath import UPath

from ...utils import WkImportError, is_remote_path

try:
    from imaris_ims_file_reader.ims import ims as ImsFile
except ImportError as e:
    raise WkImportError("imaris-ims-file-reader", "ims") from e


class PimsImsReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> set[str]:
        return {"ims"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 19

    def __init__(self, path: UPath) -> None:
        super().__init__()
        self.path = UPath(path)

        if is_remote_path(self.path):
            raise ValueError(
                f"Cannot open IMS file from {self.path}. The path must be a local file path."
            )

        # Open once to read metadata, then close — ImsFile holds an h5py file handle
        # which is not pickleable, so we must not retain it as an attribute.
        _ims = ImsFile(str(self.path), squeeze_output=False)
        t, c, z, y, x = _ims.shape
        self._file_shape: tuple[int, int, int, int, int] = (t, c, z, y, x)
        self._dtype: np.dtype = np.dtype(_ims.dtype)
        _ims.close()

        if t > 1:
            self._init_axis("t", t)
        if c > 1:
            self._init_axis("c", c)
        self._init_axis("z", z)
        self._init_axis("y", y)
        self._init_axis("x", x)

        if c > 1:
            self._register_get_frame(self.get_frame_2D, "cyx")
        else:
            self._register_get_frame(self.get_frame_2D, "yx")

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        t = ind.get("t", 0)
        z = ind.get("z", 0)
        _, c, _, y, x = self._file_shape

        # Reopen per call so no h5py handle crosses a multiprocessing boundary.
        with h5py.File(str(self.path), "r") as hf:
            dataset = hf["DataSet"]
            if "c" in self.bundle_axes:
                frames = []
                for ci in range(c):
                    loc = f"ResolutionLevel 0/TimePoint {t}/Channel {ci}/Data"
                    frames.append(dataset[loc][z])
                return np.stack(frames, axis=0)
            else:
                ci = ind.get("c", 0)
                loc = f"ResolutionLevel 0/TimePoint {t}/Channel {ci}/Data"
                return np.array(dataset[loc][z])

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._file_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        _, c, _, y, x = self._file_shape
        if c > 1:
            return (c, y, x)
        return (y, x)


def copy_ims_chunk_to_view(
    bbox: object,
    *,
    path: UPath,
    mag_view: object,
    timepoint: int,
    channel: int | None,
    num_channels: int,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    swap_xy: bool,
    dtype: DTypeLike | None,
) -> tuple[tuple[int, int], int]:
    """
    Write one shard-sized 3D block of an IMS file directly to mag_view.

    Reads exactly the 3D block for the given bounding box in a single h5py call
    per channel, then calls mag_view.write() directly — no BufferedSliceWriter
    needed. This avoids reading wasted y/x data at border shards and eliminates
    per-slice overhead.

    The signature matches the partial applied over shard-aligned args in
    add_layer_from_images, replacing PimsImages.copy_to_view for IMS files.
    """
    from ...geometry.bounding_box import BoundingBox
    from ...geometry.nd_bounding_box import NDBoundingBox
    from ..layer.view import MagView

    assert isinstance(mag_view, MagView)
    assert isinstance(bbox, (BoundingBox, NDBoundingBox))

    relative_bbox = bbox.offset(-mag_view.bounding_box.topleft)
    x_start, x_end = relative_bbox.get_bounds("x")
    y_start, y_end = relative_bbox.get_bounds("y")
    z_start, z_end = relative_bbox.get_bounds("z")

    channels_to_read = list(range(num_channels)) if channel is None else [channel]

    with h5py.File(str(path), "r") as hf:
        ds = hf["DataSet"]
        slabs = []
        for ci in channels_to_read:
            loc = f"ResolutionLevel 0/TimePoint {timepoint}/Channel {ci}/Data"
            # Read exactly the 3D block needed — one decompression per HDF5 chunk row
            slabs.append(ds[loc][z_start:z_end, y_start:y_end, x_start:x_end])
        block = np.stack(slabs, axis=0)  # (c, z, y, x)

    # Apply flips in HDF5 axis order (c, z, y, x).
    # flip_x/-y follow PimsImages convention: flip_x flips memory axis -2 (y in HDF5),
    # flip_y flips memory axis -1 (x in HDF5).
    if flip_z:
        block = block[:, ::-1]
    if flip_x:
        block = block[:, :, ::-1]
    if flip_y:
        block = block[:, :, :, ::-1]

    # Transpose to mag_view.write() convention (c, x, y, z).
    # swap_xy=False (default): (c, z, y, x) → (c, x, y, z)
    # swap_xy=True:            (c, z, y, x) → (c, y, x, z)
    block = block.transpose(0, 3, 2, 1) if not swap_xy else block.transpose(0, 2, 3, 1)

    if dtype is not None:
        block = block.astype(dtype)

    max_value = int(block.max())
    if num_channels == 1:
        block = block[0]  # (x, y, z) — single-channel layers have no c axis

    mag_view.write(block, absolute_bounding_box=bbox)

    y_size = y_end - y_start
    x_size = x_end - x_start
    return (y_size, x_size), max_value

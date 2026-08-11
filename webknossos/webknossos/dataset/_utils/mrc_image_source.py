from __future__ import annotations

import numpy as np
from upath import UPath

from ...utils import WkImportError, is_remote_path
from ..errors import CorruptImageError, UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource, register_chunked_image_source

try:
    import mrcfile
except ImportError as e:
    raise WkImportError("mrcfile", "mrcfile") from e


@register_chunked_image_source
class MrcImageSource(ChunkedImageSource):
    """
    ChunkedImageSource for MRC files. MRC data is stored as a single
    contiguous array (no internal chunking, unlike HDF5-based formats), so
    shard-sized blocks are read straight out of mrcfile's memory-mapped array
    and written to mag_view.

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

        try:
            mrc_context = mrcfile.mmap(str(path), mode="r", permissive=True)
        except Exception as e:
            # Even in permissive mode mrcfile raises on a header it cannot make
            # sense of at all, with a multi-line message aimed at developers.
            raise CorruptImageError(
                f"Cannot open MRC file {path}. "
                + "The file is likely corrupted or not a valid MRC file.",
                path=path,
            ) from e

        with mrc_context as mrc:
            if mrc.data is None:
                raise CorruptImageError(
                    f"Cannot open MRC file {path}. "
                    + "The file is likely corrupted or not a valid MRC file.",
                    path=path,
                )
            self.dtype: np.dtype = mrc.data.dtype
            ndim = mrc.data.ndim
            if ndim == 3:
                self._z, self._y, self._x = mrc.data.shape
            elif ndim == 2:
                self._z = 1
                self._y, self._x = mrc.data.shape
            else:
                raise UnsupportedImageDataError(
                    f"Unsupported MRC data dimensionality: {ndim}. "
                    "Only 2D and 3D MRC files are supported.",
                    path=path,
                )

        if 0 in (self._z, self._y, self._x):
            # In permissive mode mrcfile only warns about a header it cannot
            # parse and reports an empty extent, which would otherwise convert
            # into an empty layer or fail much later with an unrelated error.
            raise CorruptImageError(
                f"Cannot open MRC file {path}. It describes an empty image "
                + f"({self._x}×{self._y}×{self._z}), so the file is likely "
                + "corrupted or not a valid MRC file.",
                path=path,
            )

        self.num_channels = 1

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        return None

    def _read_source_box(
        self,
        *,
        timepoint: int,  # noqa: ARG002 - MRC files have no time dimension
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        with mrcfile.mmap(str(self.path), mode="r", permissive=True) as mrc:
            if mrc.data.ndim == 3:
                block = np.array(mrc.data[z, y, x])
            else:
                # add a size-1 z axis to unify with the 3D case
                block = np.array(mrc.data[y, x])[np.newaxis]
        return block[np.newaxis]  # (z, y, x) -> (c, z, y, x)

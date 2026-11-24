from __future__ import annotations

from typing import Any

import numpy as np
from upath import UPath

from ...utils import WkImportError
from .._utils.contiguous_array import read_box
from ..errors import CorruptImageError, UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions
from .image_source_registry import register_chunked_image_source

try:
    import mrcfile
    from mrcfile.mrcinterpreter import MrcInterpreter
except ImportError as e:
    raise WkImportError("mrcfile", "mrcfile") from e


class _MrcHeaderReader(MrcInterpreter):
    """Reads the header of an MRC stream, never writing back to it (which
    `MrcInterpreter` otherwise does when it is garbage-collected)."""

    def __init__(self, iostream: Any) -> None:
        super().__init__(permissive=True)
        self._read_only = True
        self._iostream = iostream
        self._read(header_only=True)


@register_chunked_image_source
class MrcImageSource(ChunkedImageSource):
    """
    ChunkedImageSource for MRC files. MRC data is stored as a single
    contiguous array (no internal chunking, unlike HDF5-based formats), so
    shard-sized blocks are read straight out of the file with seeks, which
    works for remote paths too, and written to mag_view.

    MRC files have neither channels nor timepoints, so num_channels is
    always 1 and get_layer_split_options() always returns None.
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"mrc", "rec", "st", "map", "ali"}

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)

        corrupt_error = CorruptImageError(
            f"Cannot open MRC file {path}. "
            + "The file is likely corrupted or not a valid MRC file.",
            path=path,
        )
        try:
            with path.open("rb") as file:
                header = _MrcHeaderReader(file).header
            self.dtype: np.dtype = mrcfile.utils.data_dtype_from_header(header)
            shape = mrcfile.utils.data_shape_from_header(header)
            self._data_offset = header.nbytes + int(header.nsymbt)
            file_size = path.stat().st_size
        except Exception as e:
            # Even in permissive mode mrcfile raises on a header it cannot make
            # sense of at all, with a multi-line message aimed at developers.
            raise corrupt_error from e

        if file_size < self._data_offset + self.dtype.itemsize * int(np.prod(shape)):
            raise corrupt_error

        if len(shape) == 3:
            self._z, self._y, self._x = shape
        elif len(shape) == 2:
            self._z = 1
            self._y, self._x = shape
        else:
            raise UnsupportedImageDataError(
                f"Unsupported MRC data dimensionality: {len(shape)}. "
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

    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        return None

    def _read_source_box(
        self,
        *,
        timepoint: int,  # noqa: ARG002 - MRC files have no time dimension
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        with self.path.open("rb") as file:
            block = read_box(
                file,
                offset=self._data_offset,
                shape=(self._z, self._y, self._x),
                dtype=self.dtype,
                order="C",
                box=(z, y, x),
            )
        return block[np.newaxis]  # (z, y, x) -> (c, z, y, x)

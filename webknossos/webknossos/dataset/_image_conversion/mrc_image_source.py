from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from upath import UPath

from .._utils.contiguous_array import read_box
from ..errors import CorruptImageError, UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions
from .image_source_registry import register_chunked_image_source

_HEADER_SIZE = 1024
_MODE_TO_DTYPE = {0: "i1", 1: "i2", 2: "f4", 4: "c8", 6: "u2", 12: "f2"}


class _MrcHeader(NamedTuple):
    dtype: np.dtype
    shape: tuple[int, ...]
    data_offset: int


def _parse_header(header: bytes) -> _MrcHeader:
    """Parses the parts of an MRC2014 header needed to read its data. Raises
    ValueError on a header that does not describe readable data."""
    if len(header) < _HEADER_SIZE:
        raise ValueError("The file is too short for an MRC header.")

    # Little-endian unless the machine stamp says otherwise.
    byte_order = ">" if header[212:214] == b"\x11\x11" else "<"
    words = np.frombuffer(header, dtype=f"{byte_order}i4", count=24)
    if int(words[3]) not in _MODE_TO_DTYPE:
        # Some files in the wild carry a machine stamp of the wrong byte order.
        byte_order = ">" if byte_order == "<" else "<"
        words = words.byteswap()
    nx, ny, nz, mode = (int(i) for i in words[:4])
    mz, ispg, nsymbt = int(words[9]), int(words[22]), int(words[23])

    if mode not in _MODE_TO_DTYPE:
        raise ValueError(f"Unknown MRC mode {mode}.")
    if min(nx, ny, nz, nsymbt) < 0:
        raise ValueError("Negative MRC dimensions or extended header size.")

    if 401 <= ispg <= 630:
        # A stack of volumes, each `mz` sections deep.
        if mz <= 0:
            raise ValueError(f"Invalid MRC volume depth {mz}.")
        shape: tuple[int, ...] = (nz // mz, mz, ny, nx)
    elif ispg == 0 and nz == 1:
        shape = (ny, nx)
    else:
        shape = (nz, ny, nx)
    return _MrcHeader(
        dtype=np.dtype(byte_order + _MODE_TO_DTYPE[mode]),
        shape=shape,
        data_offset=_HEADER_SIZE + nsymbt,
    )


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
                header = _parse_header(file.read(_HEADER_SIZE))
            file_size = path.stat().st_size
        except ValueError as e:
            raise corrupt_error from e
        self._file_dtype = header.dtype
        self.dtype: np.dtype = header.dtype.newbyteorder("=")
        self._data_offset = header.data_offset
        shape = header.shape

        if file_size < self._data_offset + self.dtype.itemsize * math.prod(shape):
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
            # Would otherwise convert into an empty layer or fail much later
            # with an unrelated error.
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
                dtype=self._file_dtype,
                order="C",
                box=(z, y, x),
            )
        return block.astype(self.dtype, copy=False)[
            np.newaxis
        ]  # (z, y, x) -> (c, z, y, x)

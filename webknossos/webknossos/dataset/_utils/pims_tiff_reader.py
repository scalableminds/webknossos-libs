import json
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pims import FramesSequenceND

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "Cannot import tifffile, please install it e.g. using 'webknossos[tifffile]'"
    ) from e


# This indexing function is adapted from zarr-python to work with tiffile's aszarr function
# See https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/indexing.py
class _ChunkProjection(NamedTuple):
    chunk_coords: tuple[int, ...]
    chunk_selection: tuple[slice | int, ...]
    out_selection: tuple[slice | None, ...]


def _chunk_indexing(
    selection: tuple[slice | int, ...],
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> Iterator[_ChunkProjection]:
    from itertools import product

    class ChunkDimProjection(NamedTuple):
        dim_chunk_ix: int
        dim_chunk_sel: slice | int
        dim_out_sel: slice | None

    def ceildiv(a: int, b: int) -> int:
        return -(a // -b)

    def slice_dim_indexer(
        dim_sel: slice, dim_len: int, dim_chunk_len: int
    ) -> Iterator[ChunkDimProjection]:
        start, stop, step = dim_sel.indices(dim_len)
        assert step == 1

        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = start // dim_chunk_len
        dim_chunk_ix_to = ceildiv(stop, dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):
            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * dim_chunk_len
            dim_limit = min(dim_len, (dim_chunk_ix + 1) * dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - start) % 1
                if remainder:
                    dim_chunk_sel_start += 1 - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = dim_offset - start

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = start - dim_offset
                dim_out_offset = 0

            if stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, 1)
            dim_chunk_nitems = dim_chunk_sel_stop - dim_chunk_sel_start
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)

    def int_dim_indexer(
        dim_sel: int, dim_chunk_len: int
    ) -> Iterator[ChunkDimProjection]:
        dim_chunk_ix = dim_sel // dim_chunk_len
        dim_offset = dim_chunk_ix * dim_chunk_len
        yield ChunkDimProjection(dim_chunk_ix, dim_sel - dim_offset, None)

    # setup per-dimension indexers
    dim_indexers = [
        slice_dim_indexer(dim_sel, dim_len, dim_chunk_len)
        if isinstance(dim_sel, slice)
        else int_dim_indexer(dim_sel, dim_chunk_len)
        for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape)
    ]

    for dim_projections in product(*dim_indexers):
        chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
        chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
        out_selection = tuple(
            p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
        )

        yield _ChunkProjection(chunk_coords, chunk_selection, out_selection)


class PimsTiffReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> set[str]:
        return {"tif", "tiff"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # We decided to use a custom reader for tiff files to support images with more than 3 dimensions out of the box.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 19

    def __init__(self, path: PathLike) -> None:
        super().__init__()

        self.path = Path(path)

        _tiff = tifffile.TiffFile(self.path).series[0]
        self._tiff_axes = tuple(_tiff.axes.lower())  # All the axes in the tiff file
        for axis, shape in zip(self._tiff_axes, _tiff.shape):
            self._init_axis(axis, shape)

        self._tiff_shape = _tiff.shape

        # Selecting the first page to get the dtype and shape
        if hasattr(_tiff, "pages"):
            _tmp = _tiff.pages[0]
        else:
            _tmp = _tiff["pages"][0]  # type: ignore
        assert _tmp is not None, "No pages found in tiff file."
        self._dtype = _tmp.dtype or np.dtype("uint8")
        self._shape = _tmp.shape
        self._other_axes = tuple(
            axis for axis in self._tiff_axes if axis not in _tmp.axes.lower()
        )  # Axes that are not present in a single tiff page

        if "c" in self._tiff_axes:
            self._register_get_frame(self.get_frame_2D, "cyx")
        else:
            self._register_get_frame(self.get_frame_2D, "yx")

        expected_page_count = int(
            np.prod([self.sizes[axis] for axis in self._other_axes])
        )
        self._page_mode = len(_tiff.pages) == expected_page_count

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        _tiff = tifffile.TiffFile(self.path).series[0]

        # We are using aszarr because it provides a chunked interface
        # to the tiff file's content. However, we don't want to add
        # zarr-python as a dependency. So we just implement the indexing
        # ourselves and rely on the fact that tifffile isn't using more
        # complex zarr features such as compressors, filters, F-order, fillvalue etc.
        zarr_store = _tiff.aszarr(
            level=0
        )  # for multi-scale tiffs, we use the highest resolution
        zarray = json.loads(zarr_store[".zarray"])

        assert zarray["zarr_format"] == 2
        assert zarray["order"] == "C"
        assert np.dtype(zarray["dtype"]) == self._dtype
        assert zarray.get("compressor") is None
        assert zarray.get("filters") in (None, [])
        assert zarray["fill_value"] == 0
        array_shape = tuple(zarray["shape"])
        chunk_shape = tuple(zarray["chunks"])

        # Prepare output array for this frame
        out_shape = tuple(self.sizes[axis] for axis in self.bundle_axes)
        out = np.zeros(out_shape, dtype=self._dtype)

        # Axes that need to be broadcasted from page to output
        broadcast_axes = tuple(
            axis
            for axis in self._tiff_axes
            if axis in self.bundle_axes and axis not in self._other_axes
        )

        # Prepare selection of the data to read for this frame
        selection: tuple[slice | int, ...] = tuple(
            slice(None) if axis in broadcast_axes else ind[axis]
            for axis in self._tiff_axes
        )

        for chunk_projection in _chunk_indexing(selection, array_shape, chunk_shape):
            # read data from zarr store
            chunk_data = (
                zarr_store[".".join(map(str, chunk_projection.chunk_coords))]
                .ravel()
                .reshape(chunk_shape)
            )
            # write in output array
            out[chunk_projection.out_selection] = chunk_data[
                chunk_projection.chunk_selection
            ]

        return out

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tiff_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self._shape

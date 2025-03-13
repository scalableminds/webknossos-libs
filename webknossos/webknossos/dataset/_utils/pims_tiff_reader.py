import json
from os import PathLike
from pathlib import Path
from typing import Iterator, NamedTuple, Set, Tuple, Union

import numpy as np
from pims import FramesSequenceND

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "Cannot import tifffile, please install it e.g. using 'webknossos[tifffile]'"
    ) from e

ChunkCoords = tuple[int, ...]
Selection = tuple[slice, ...]


class _ChunkDimProjection(NamedTuple):
    dim_chunk_ix: int
    dim_chunk_sel: Union[slice, int]
    dim_out_sel: Union[slice, None]


def _ceildiv(a, b):
    from math import ceil

    return ceil(a / b)


class _SliceDimIndexer:
    dim_sel: slice
    dim_len: int
    dim_chunk_len: int

    start: int
    stop: int

    def __init__(self, dim_sel: slice, dim_len: int, dim_chunk_len: int):
        self.start, self.stop, step = dim_sel.indices(dim_len)
        assert step == 1

        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len

    def __iter__(self) -> Iterator[_ChunkDimProjection]:
        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = self.start // self.dim_chunk_len
        dim_chunk_ix_to = _ceildiv(self.stop, self.dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):
            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_limit = min(self.dim_len, (dim_chunk_ix + 1) * self.dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if self.start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - self.start) % 1
                if remainder:
                    dim_chunk_sel_start += 1 - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = dim_offset - self.start

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.start - dim_offset
                dim_out_offset = 0

            if self.stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, 1)
            dim_chunk_nitems = dim_chunk_sel_stop - dim_chunk_sel_start
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class _IntDimIndexer:
    dim_sel: int
    dim_len: int
    dim_chunk_len: int

    def __init__(self, dim_sel: int, dim_len: int, dim_chunk_len: int) -> None:
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len

    def __iter__(self) -> Iterator[_ChunkDimProjection]:
        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield _ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class _ChunkProjection(NamedTuple):
    chunk_coords: ChunkCoords
    chunk_selection: tuple[Union[slice, int], ...]
    out_selection: tuple[Union[slice, None], ...]


class BasicIndexer:
    dim_indexers: list[_SliceDimIndexer]
    shape: ChunkCoords

    def __init__(
        self,
        selection: tuple[Union[slice, int], ...],
        shape: ChunkCoords,
        chunk_shape: ChunkCoords,
    ):
        # setup per-dimension indexers
        self.dim_indexers = [
            _SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)
            if isinstance(dim_sel, slice)
            else _IntDimIndexer(dim_sel, dim_len, dim_chunk_len)
            for dim_sel, dim_len, dim_chunk_len in zip(selection, shape, chunk_shape)
        ]

    def __iter__(self) -> Iterator[_ChunkProjection]:
        from itertools import product

        for dim_projections in product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield _ChunkProjection(chunk_coords, chunk_selection, out_selection)


class PimsTiffReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> Set[str]:
        return {"tif", "tiff"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # We decided to use a custom reader for tiff files to support images with more than 3 dimensions out of the box.
    # Default is 10, and bioformats priority is 2.
    # Our custom reader for imagej_tiff has priority 20.
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
        zarr_store = _tiff.aszarr()
        zarray = json.loads(zarr_store[".zarray"])

        out_shape = tuple(self.sizes[axis] for axis in self.bundle_axes)
        out = np.zeros(out_shape, dtype=self._dtype)

        # Axes that need to be broadcasted from page to output
        broadcast_axes = tuple(
            axis
            for axis in self._tiff_axes
            if axis in self.bundle_axes and axis not in self._other_axes
        )

        selector_list: list[Union[slice, int]] = []
        for axis in self._tiff_axes:
            if axis in broadcast_axes:
                selector_list.append(slice(None))  # broadcast
            else:
                selector_list.append(ind[axis])
        selector = tuple(selector_list)
        print(self._tiff_axes, self.sizes, selector)
        array_shape = tuple(zarray["shape"])
        chunk_shape = tuple(zarray["chunks"])
        for chunk_proj in BasicIndexer(selector, array_shape, chunk_shape):
            print(chunk_proj)
            chunk_data = (
                zarr_store[".".join(map(str, chunk_proj.chunk_coords))]
                .ravel()
                .reshape(chunk_shape)
            )
            chunk_slice = chunk_data[chunk_proj.chunk_selection]
            print(chunk_slice.shape)
            out[chunk_proj.out_selection] = chunk_slice

        return out

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tiff_shape

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

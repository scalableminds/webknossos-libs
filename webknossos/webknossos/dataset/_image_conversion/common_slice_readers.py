"""Readers for common 2D raster images, and for sequences of image files.

Decoding is delegated to `imageio`; these classes only handle which files to
read, in what order, and how to present them as one sequence of slices.
"""

from __future__ import annotations

import fnmatch
import glob
import os
import zipfile
from collections.abc import Callable, Iterable
from io import BytesIO
from typing import Any
from warnings import warn

import numpy as np
from imageio import v2 as iio
from natsort import natsort_keygen
from numpy.typing import DTypeLike

from ..errors import UnsupportedImageDataError
from .image_source_registry import register_slice_reader
from .slice_reader import SliceReader

_natsort_key = natsort_keygen()


def imread(uri: Any, **kwargs: Any) -> np.ndarray:
    """Reads one image file into an array, stripping imageio's metadata."""
    return np.asarray(iio.imread(uri, **kwargs))


def _plane_axes(plane: np.ndarray, source: object) -> str:
    """The axes of one decoded raster image, always `(y, x)` or `(y, x, c)`.

    Safe to state rather than guess: the only axis that could be mistaken for
    another is the file index, which belongs to the reader, not to the array.
    """
    if plane.ndim == 2:
        return "yx"
    if plane.ndim == 3:
        return "yxc"
    raise UnsupportedImageDataError(
        f"Got a {plane.ndim}-dimensional image from {source}, but a raster "
        + "image must decode to (y, x) or (y, x, channels)."
    )


@register_slice_reader
class SingleImageSliceReader(SliceReader):
    """Reads a single 2D raster image into a length-1 sequence."""

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"png", "jpg", "jpeg", "gif", "bmp", "ico"}

    class_priority = 12

    # A three-channel png is an RGB photograph, not three acquisitions.
    channels_are_rgb = True

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__()
        self._data = imread(filename, **kwargs)
        axes = _plane_axes(self._data, filename)
        for axis, size in zip(axes, self._data.shape):
            self._init_axis(axis, size)
        self._set_get_slice(self._read_image, axes)

    def _read_image(self, **coords: int) -> np.ndarray:
        del coords  # there is only one image
        return self._data

    @property
    def pixel_type(self) -> DTypeLike:
        return self._data.dtype


def _collect_files(
    path_spec: str | Iterable[str],
) -> tuple[list[str], zipfile.ZipFile | None]:
    """Resolves a directory, glob pattern, zip archive or iterable of paths
    into the list of files to read, in the order they should be read. An
    explicitly passed list is returned as given, preserving a custom order;
    callers that want it sorted anyway do so themselves."""
    if not isinstance(path_spec, str):
        return [str(i) for i in path_spec], None

    if zipfile.is_zipfile(path_spec):
        archive = zipfile.ZipFile(path_spec, "r")
        filepaths = [fn for fn in archive.namelist() if fnmatch.fnmatch(fn, "*.*")]
        filepaths.sort(key=_natsort_key)
        return filepaths, archive

    if os.path.isdir(path_spec):
        warn(
            "Loading ALL files in this directory. To ignore extraneous files, "
            "use a pattern like 'path/to/images/*.png'",
            UserWarning,
            stacklevel=2,
        )
        filepaths = [
            os.path.abspath(os.path.join(path_spec, filename))
            for filename in os.listdir(path_spec)
        ]
    else:
        filepaths = glob.glob(path_spec)

    filepaths.sort(key=_natsort_key)
    if not filepaths:
        raise OSError(f"No files were found matching {path_spec}.")
    return filepaths, None


class MultiImageSliceReader(SliceReader):
    """Reads a directory, glob pattern, zip archive or list of 2D image files
    as one sequence, each file contributing one slice along `z`. Every file
    must decode to the shape of the first, which is the only one read up front.
    """

    def __init__(self, path_spec: str | Iterable[str], **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        # Set before anything that can raise, so __del__ -> close() does not
        # fail with AttributeError when __init__ aborts.
        self._zipfile: zipfile.ZipFile | None = None
        self._filepaths, self._zipfile = _collect_files(path_spec)
        first_slice = self.imread(self._filepaths[0], **self.kwargs)
        self._dtype = first_slice.dtype

        # imageio also decodes formats this class is not the preferred reader
        # for — a directory of TIFFs ends up here, where a "c" axis is separate
        # acquisitions rather than RGB. The files say which case this is.
        first_extension = os.path.splitext(self._filepaths[0])[1].lstrip(".").lower()
        self.channels_are_rgb = (
            first_extension in SingleImageSliceReader.supported_file_extensions()
        )

        self._init_axis("z", len(self._filepaths))
        plane_axes = _plane_axes(first_slice, self._filepaths[0])
        for axis, size in zip(plane_axes, first_slice.shape):
            self._init_axis(axis, size)
        # The declared axes are one file's; `z` picks the file, reaching
        # _read_file through the coords every get_slice call carries.
        self._set_get_slice(self._read_file, plane_axes)
        self.iter_axes = ["z"]

    def imread(self, filename: str, **kwargs: Any) -> np.ndarray:
        if self._zipfile is not None:
            return imread(BytesIO(self._zipfile.read(filename)), **kwargs)
        return imread(filename, **kwargs)

    def _read_file(self, **coords: int) -> np.ndarray:
        return self.imread(self._filepaths[coords["z"]], **self.kwargs)

    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None
        super().close()

    def __del__(self) -> None:
        self.close()

    @property
    def pixel_type(self) -> DTypeLike:
        return self._dtype


class StackedFileSliceReader(SliceReader):
    """Stacks several n-dimensional image files along one added axis (`t` by
    default) that selects the file. Each is opened with the same reader class
    and must expose identical axes and sizes."""

    def __init__(
        self,
        path_spec: str | Iterable[str],
        # Any rather than SliceReader: not a fixed type since it can be a
        # reader class or a factory function.
        reader_cls: Callable[..., Any] | None = None,
        axis_name: str = "t",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self._zipfile: zipfile.ZipFile | None = None
        if reader_cls is None:
            from .image_source_registry import open_slice_reader

            self.reader_cls: Callable[..., Any] = open_slice_reader
        else:
            self.reader_cls = reader_cls
        self._filepaths, self._zipfile = _collect_files(path_spec)
        if not isinstance(path_spec, str):
            # Unlike MultiImageSliceReader, an explicit list is sorted here
            # too, rather than kept in the caller's order.
            self._filepaths.sort(key=_natsort_key)

        with self.reader_cls(self._filepaths[0], **self.kwargs) as reader:
            for ax in reader.axes:
                self._init_axis(ax, reader.sizes[ax])
            self._pixel_type = reader.pixel_type
        self._imseq_axis = axis_name
        self._init_axis(axis_name, len(self._filepaths))
        self.iter_axes = [axis_name]

    @property
    def bundle_axes(self) -> list[str]:
        return self._bundle_axes[:]

    @bundle_axes.setter
    def bundle_axes(self, value: Iterable[str]) -> None:
        """Overrides the base class' adapter, because `_get_seq_slice` defers
        the axis handling to the per-file reader."""
        value = list(value)
        if invalid := [k for k in value if k not in self._sizes]:
            raise ValueError(f"axes {invalid!r} do not exist")
        if self._imseq_axis in value:
            raise ValueError("The sequence axis cannot be bundled.")

        for k in value:
            if k in self._iter_axes:
                self._iter_axes.remove(k)
        self._bundle_axes = value
        self._get_slice_wrapped = self._get_seq_slice

    def _get_seq_slice(self, **coords: int) -> np.ndarray:
        i = coords.pop(self._imseq_axis)
        with self.reader_cls(self._filepaths[i], **self.kwargs) as reader:
            # Check whether this file matches the shape of the first one.
            for ax in self.sizes:
                if ax == self._imseq_axis:
                    continue
                if ax not in reader.sizes:
                    raise RuntimeError(f"{self._filepaths[i]} does not have axis {ax}")
                if reader.sizes[ax] != self.sizes[ax]:
                    raise RuntimeError(
                        f"In {self._filepaths[i]}, the size of axis {ax} was unexpected"
                    )
            reader.bundle_axes = self.bundle_axes
            return reader._get_slice_wrapped(**coords)

    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None
        super().close()

    @property
    def pixel_type(self) -> DTypeLike:
        return self._pixel_type

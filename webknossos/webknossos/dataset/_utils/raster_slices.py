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

from .image_reader_registry import register_image_reader
from .slice_sequence import NDSliceSequence, SliceSequence

_natsort_key = natsort_keygen()


def imread(uri: Any, **kwargs: Any) -> np.ndarray:
    """Reads one image file into an array, stripping imageio's metadata."""
    return np.asarray(iio.imread(uri, **kwargs))


@register_image_reader
class SingleImageSlices(SliceSequence):
    """Reads a single 2D raster image into a length-1 sequence.

    Deliberately a flat `SliceSequence` rather than an `NDSliceSequence`:
    `SlicedImageSource` derives the axis order of such images from the shape alone,
    which is how these formats have always been handled.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return {"png", "jpg", "jpeg", "gif", "bmp", "ico"}

    class_priority = 12

    def __init__(self, filename: str, **kwargs: Any) -> None:
        self._data = imread(filename, **kwargs)

    def __len__(self) -> int:
        return 1

    def get_slice(self, i: int) -> np.ndarray:
        del i  # there is only one slice
        return self._data

    @property
    def pixel_type(self) -> DTypeLike:
        return self._data.dtype

    @property
    def slice_shape(self) -> tuple[int, ...]:
        return self._data.shape


def _collect_files(
    path_spec: str | Iterable[str], sort_explicit_lists: bool
) -> tuple[list[str], zipfile.ZipFile | None]:
    """Resolves a directory, glob pattern, zip archive or iterable of paths
    into the list of files to read, in the order they should be read."""
    if not isinstance(path_spec, str):
        filepaths = [str(i) for i in path_spec]
        # Deliberate asymmetry: StackedFileSlices re-sorts an explicitly
        # passed list, MultiImageSlices does not. Both orders are long-standing,
        # and changing either would silently reorder z in existing conversions.
        # The cost is that for StackedFileSlices a custom `z_slices_sort_key`
        # given to `Dataset.from_images` does not survive.
        if sort_explicit_lists:
            filepaths.sort(key=_natsort_key)
        return filepaths, None

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


class MultiImageSlices(SliceSequence):
    """Reads a directory, glob pattern, zip archive or list of 2D image files
    as one sequence, with each file contributing one slice."""

    def __init__(self, path_spec: str | Iterable[str], **kwargs: Any) -> None:
        self.kwargs = kwargs
        # Set before anything that can raise, so __del__ -> close() does not
        # fail with AttributeError when __init__ aborts.
        self._zipfile: zipfile.ZipFile | None = None
        self._filepaths, self._zipfile = _collect_files(
            path_spec, sort_explicit_lists=False
        )
        first_slice = self.imread(self._filepaths[0], **self.kwargs)
        self._first_slice_shape = first_slice.shape
        self._dtype = first_slice.dtype

    def imread(self, filename: str, **kwargs: Any) -> np.ndarray:
        if self._zipfile is not None:
            return imread(BytesIO(self._zipfile.read(filename)), **kwargs)
        return imread(filename, **kwargs)

    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None
        super().close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._filepaths)

    def get_slice(self, i: int) -> np.ndarray:
        return self.imread(self._filepaths[i], **self.kwargs)

    @property
    def slice_shape(self) -> tuple[int, ...]:
        return self._first_slice_shape

    @property
    def pixel_type(self) -> DTypeLike:
        return self._dtype


class StackedFileSlices(NDSliceSequence):
    """Stacks several n-dimensional image files along one added axis.

    Each file is opened with the same reader class and must expose identical
    axes and sizes; the added axis (`t` by default) selects the file.
    """

    def __init__(
        self,
        path_spec: str | Iterable[str],
        # Any rather than NDSliceSequence: callers pass the class that opened
        # the first file, which is only known to be a SliceSequence. The
        # isinstance check below is what actually enforces the requirement.
        reader_cls: Callable[..., Any] | None = None,
        axis_name: str = "t",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self._zipfile: zipfile.ZipFile | None = None
        if reader_cls is None:
            from .image_reader_registry import open_images

            self.reader_cls: Callable[..., Any] = open_images
        else:
            self.reader_cls = reader_cls
        self._filepaths, self._zipfile = _collect_files(
            path_spec, sort_explicit_lists=True
        )

        with self.reader_cls(self._filepaths[0], **self.kwargs) as reader:
            if not isinstance(reader, NDSliceSequence):
                raise ValueError("Reader is not a subclass of NDSliceSequence")
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
        """Overrides the base class' resolver, because `_get_seq_slice` defers
        the axis bundling to the per-file reader."""
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

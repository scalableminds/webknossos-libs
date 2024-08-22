import tempfile
from os import PathLike
from pathlib import Path
from typing import Set, Tuple

import numpy as np
from pims import FramesSequenceND

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "Cannot import tifffile, please install it e.g. using 'webknossos[tifffile]'"
    ) from e


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

        path = Path(path)

        _tiff = tifffile.TiffFile(path).series[0]
        self._tiff_axes = tuple(_tiff.axes.lower())
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
        try:
            self._memmap = tifffile.memmap(path, mode="r")
        except ValueError:
            # If the tiff file is not memory-mappable, we create a temporary file and copy the data to it.
            with tempfile.TemporaryFile() as tmp:
                self._memmap = np.memmap(
                    tmp,
                    dtype=self._dtype,
                    shape=_tiff.shape,
                    mode="w+",
                )
            # The axes per page are a subset of the axes of the tifffile. We copy the data from the tifffile to the memmap with correct axes.
            # While the actual axes are e.g. ["t", "z", "y", "x"], with a shape like (3, 5, 100, 200), the axes of the tifffile consist of the axes of a singe page, e.g. ["y", "x"]. And the number of pages is 15.
            # We have to iterate over the pages and copy the data to the correct position in the memmap.
            for i in range(len(_tiff.pages)):
                other_axes = tuple(
                    axis for axis in self._tiff_axes if axis not in _tmp.axes.lower()
                )
                slices = []
                for j, axis in enumerate(other_axes):
                    size = self.sizes[axis]
                    index = (
                        i
                        // (
                            np.prod(
                                [self.sizes[axis] for axis in other_axes[j + 1 :]],
                                dtype=int,
                            )
                        )
                        % size
                    )
                    slices.append(slice(index, index + 1))

                _tiff.asarray(key=i, out=self._memmap[tuple(slices)])
        if "c" in self._tiff_axes:
            self._register_get_frame(self.get_frame_2D, "cyx")
        else:
            self._register_get_frame(self.get_frame_2D, "yx")

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        # A frame of the tiff file might have less axes than the desired shape of a frame in the FramesSequenceND.
        # To get the desired axes we need to iterate over the axes of the FramesSequenceND and extract the data from the tiff file.
        slice_tuple = tuple(
            slice(None) if axis in self.bundle_axes else slice(ind[axis], ind[axis] + 1)
            for axis in self._tiff_axes
        )

        data = self._memmap[slice_tuple]
        data = np.moveaxis(
            data,
            tuple(self._tiff_axes.index(axis) for axis in self.bundle_axes),
            range(len(self.bundle_axes)),
        )
        return data.squeeze(axis=tuple(range(len(self.bundle_axes), len(data.shape))))

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tiff_shape

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

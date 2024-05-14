from os import PathLike
from pathlib import Path
from typing import Set, Tuple

import numpy as np

try:
    from pims import FramesSequenceND
except ImportError as e:
    raise RuntimeError(
        "Cannot import pims, please install it e.g. using 'webknossos[all]'"
    ) from e

try:
    import tifffile
except ImportError as e:
    raise RuntimeError(
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
        # Selecting the first page to get the dtype and shape
        if hasattr(_tiff, "pages"):
            _tmp = _tiff.pages[0]
        else:
            _tmp = _tiff["pages"][0]
        self._dtype = _tmp.dtype
        self._shape = _tmp.shape
        self._tiff_axes = tuple(_tiff.axes.lower())

        self._memmap = tifffile.memmap(
            path,
            mode="r",
        )

        self._register_get_frame(self.get_frame_2D, _tmp.axes.lower())

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
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

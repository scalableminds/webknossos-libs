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
        self._tiff = tifffile.TiffFile(path).series[0]

        for axis, shape in zip(self._tiff.get_axes(), self._tiff.get_shape()):
            self._init_axis(axis.lower(), shape)

        if hasattr(self._tiff, "pages"):
            tmp = self._tiff.pages[0]
        else:
            tmp = self._tiff["pages"][0]
        self._dtype = tmp.dtype
        self._shape = tmp.shape
        self._register_get_frame(self.get_frame, tmp.axes.lower())

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    def get_frame(self, ind: int) -> np.ndarray:
        data = self._tiff.asarray(key=ind)
        return data

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

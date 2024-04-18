from os import PathLike
from pathlib import Path
from typing import Set

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
        tiff = tifffile.TiffFile(path)
        self.memmap = tiff.asarray(out="memmap")

        for axis, shape in zip(tiff.series[0].get_axes(), tiff.series[0].get_shape()):
            self._init_axis(axis.lower(), shape)

        if "c" in self.axes:
            self._register_get_frame(self._get_frame, "cyx")
        else:
            self._register_get_frame(self._get_frame, "yx")

    @property
    def pixel_type(self) -> np.dtype:
        return self.memmap.dtype

    def _get_frame(self, **ind: int) -> np.ndarray:
        return self.memmap[tuple(ind[axis] for axis in self._iter_axes)]

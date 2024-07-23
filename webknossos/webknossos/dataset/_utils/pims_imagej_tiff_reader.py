from os import PathLike
from pathlib import Path
from typing import Set

import numpy as np
from pims import FramesSequenceND

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "Cannot import tifffile, please install it e.g. using 'webknossos[tifffile]'"
    ) from e


class PimsImagejTiffReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> Set[str]:
        return {"tif", "tiff"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 20

    def __init__(self, path: PathLike) -> None:
        super().__init__()
        path = Path(path)
        tiff = tifffile.TiffFile(path)
        assert tiff.is_imagej, f"{path} is not an ImageJ Tiff"
        channels = tiff.imagej_metadata["channels"]  # type: ignore
        z = tiff.imagej_metadata["images"] / channels  # type: ignore

        self.memmap = tifffile.memmap(path)
        # shape should be zcyx
        assert len(self.memmap.shape) == 4
        assert self.memmap.shape[0] == z
        assert self.memmap.shape[1] == channels

        self._init_axis("x", self.memmap.shape[3])
        self._init_axis("y", self.memmap.shape[2])
        self._init_axis("z", z)
        self._init_axis("c", channels)
        self._register_get_frame(self._get_frame, "cyx")

    @property
    def pixel_type(self) -> np.dtype:
        return self.memmap.dtype

    def _get_frame(self, **ind: int) -> np.ndarray:
        return self.memmap[ind["z"]]

import logging
from os import PathLike
from typing import Set

import numpy as np
from pydicom import dcmread

try:
    from pims import FramesSequenceND
except ImportError as e:
    raise RuntimeError(
        "Cannot import pims, please install it e.g. using 'webknossos[all]'"
    ) from e

logging.getLogger(__name__).setLevel(logging.DEBUG)


class DicomReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> Set[str]:
        return {"dcm", "dicom"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 20

    def __init__(self, path: PathLike) -> None:
        super().__init__()
        self.img = dcmread(path)
        self.pixel_type = self.img.dtype

        self._init_axis("x", self.img.shape[2])
        self._init_axis("y", self.img.shape[1])
        self._init_axis("z", self.img.shape[0])
        self._init_axis("c", 1 if len(self.img.shape) == 3 else self.img.shape[3])
        self._register_get_frame(self._get_frame, "cyx")

    def _get_frame(self, **ind: int) -> np.ndarray:
        return self.img.voxel_array[ind["z"]]

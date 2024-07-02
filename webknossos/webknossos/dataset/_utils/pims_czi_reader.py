from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Dict, Iterator, List, Set

import numpy as np
from pims import FramesSequenceND

try:
    from pylibCZIrw import czi as pyczi
except ImportError as e:
    raise ImportError(
        "Cannot import pylibCZIrw, please install it e.g. using 'webknossos[czi]'"
    ) from e

PIXEL_TYPE_TO_DTYPE = {
    "Gray8": "<u1",
    "Gray16": "<u2",
    "Gray32Float": "<f4",
    "Bgr24": "<u1",
    "Bgr48": "<u2",
    "Bgr96Float": "<f4",
    "Bgra32": "<4u1",
    "Gray64ComplexFloat": "<F8",
    "Bgr192ComplexFloat": "<F8",
    "Gray32": "<i4",
    "Gray64": "<i8",
}


class PimsCziReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> Set[str]:
        return {"czi"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # Default is 10, and bioformats priority (which is the only other reader supporting czi) is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 20

    def __init__(self, path: PathLike, czi_channel: int = 0) -> None:
        self.path = Path(path)
        self.czi_channel = czi_channel
        self.axis_offsets: Dict[str, int] = {}
        super().__init__()
        with self.czi_file() as czi_file:
            for axis, (
                start,
                end,
            ) in czi_file.total_bounding_box.items():
                axis = axis.lower()
                if axis == "c":
                    continue
                length = end - start
                if axis not in "xy" and length == 1:
                    # not propagating axes of length one
                    continue
                assert length >= 0, f"axis length must be >= 0, got {length}"
                self._init_axis(axis, length)
                self.axis_offsets[axis] = start
            self._czi_pixel_type = czi_file.get_channel_pixel_type(self.czi_channel)
            if self._czi_pixel_type.startswith("Bgra"):
                self._init_axis("c", 4)
            elif self._czi_pixel_type.startswith("Bgr"):
                self._init_axis("c", 3)
            elif self._czi_pixel_type.startswith("Gray"):
                self._init_axis("c", 1)
            elif self._czi_pixel_type == "Invalid":
                raise ValueError(
                    f"czi_channel {self.czi_channel} does not exist in {self.path}"
                )
            else:
                raise ValueError(
                    f"Got unsupported czi pixel-type {self._czi_pixel_type} in {self.path}"
                )

        self._register_get_frame(self.get_frame_2D, "yxc")

    @contextmanager
    def czi_file(self) -> Iterator[pyczi.CziReader]:
        with pyczi.open_czi(str(self.path)) as czi_file:
            yield czi_file

    def available_czi_channels(self) -> List[int]:
        with self.czi_file() as czi_file:
            return sorted(czi_file.pixel_types.keys())

    @property  # potential @cached_property for py3.8+
    def pixel_type(self) -> np.dtype:
        return np.dtype(PIXEL_TYPE_TO_DTYPE[self._czi_pixel_type])

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        plane = {k.upper(): v for k, v in ind.items()}
        for axis in plane.keys():
            if axis in self.axis_offsets:
                plane[axis] += self.axis_offsets[axis]

        # safe-guard against x/y in ind argument,
        # we always read the whole slice here:
        plane.pop("X", None)
        plane.pop("Y", None)

        plane["C"] = self.czi_channel
        with self.czi_file() as czi_file:
            a = czi_file.read(plane=plane)
            num_channels = a.shape[-1]
            if num_channels == 3:
                # convert from bgr to rgb
                a = np.flip(a, axis=-1)
            elif num_channels == 4:
                # convert from bgra to rgba
                a_red = a[:, :, 2].copy(order="K")
                a[:, :, 2] = a[:, :, 0]
                a[:, :, 0] = a_red
            return a

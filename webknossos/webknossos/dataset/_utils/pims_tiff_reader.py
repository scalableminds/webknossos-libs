from os import PathLike
from pathlib import Path
from typing import Iterable, Set, Tuple

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
            # Selecting the first page to get the dtype and shape
            tmp = self._tiff.pages[0]
        else:
            tmp = self._tiff["pages"][0]
        # Updating the bundle axes of FramesSequenceND to match the metadata of the tiff file
        self._dtype = tmp.dtype
        self._shape = tmp.shape
        self._bundle_axes_page = tmp.axes.lower()
        # if len(self._tiff.axes) <= 3 or (
        #     len(self._tiff.axes) == 4 and "c" in self._tiff.axes
        # ):
        #     raise RuntimeError(
        #         "This reader is not suitable for 2D or 3D images. Use the default tiff reader."
        #     )
        self._register_get_frame(self.get_frame_2D, self._bundle_axes_page)

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    def _extract_from_tiff_page(self, **ind: int) -> np.ndarray:
        # A frame of the tiff file might have more axes than the desired shape of a frame in the FramesSequenceND.
        # To reduce the axes of the tiff file to the desired shape we need to iterate over the axes of the tiff file and extract the data.
        index_slice = tuple(
            slice(ind[axis], ind[axis] + 1)
            if axis not in self.bundle_axes
            else slice(None)
            for axis in self._bundle_axes_page
        )
        key = 0
        iter_size = 1
        for axis, other_axis in zip(self.iter_axes[-1:0:-1], self._iter_axes[-2::-1]):
            # Calculate the key for the tiff file to get the correct frame
            key += (
                ind[other_axis] * (iter_size := iter_size * self.sizes[axis])
                + ind[axis]
            )
        data = self._tiff.asarray(key=key)[index_slice]
        return data

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        # A frame of the tiff file might have less axes than the desired shape of a frame in the FramesSequenceND.
        # To get the desired axes we need to iterate over the axes of the FramesSequenceND and extract the data from the tiff file.
        desired_shape = tuple(self.sizes[axis] for axis in self.bundle_axes)
        data = np.zeros(desired_shape, dtype=self.pixel_type)

        for current_axis in self.bundle_axes:
            if current_axis not in self._bundle_axes_page:
                for i in range(self.sizes[current_axis]):
                    ind[current_axis] = i
                    tiff_page_data = self._extract_from_tiff_page(**ind).squeeze()
                    data[
                        tuple(
                            slice(i, i + 1) if current_axis == axis else slice(None)
                            for axis in self.bundle_axes
                        )
                    ] = tiff_page_data
            else:
                data = self._extract_from_tiff_page(**ind).squeeze()

        return data

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

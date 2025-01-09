from itertools import product
from os import PathLike
from pathlib import Path
from typing import Set, Tuple, Union

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

        self.path = Path(path)

        _tiff = tifffile.TiffFile(self.path).series[0]
        self._tiff_axes = tuple(_tiff.axes.lower())  # All the axes in the tiff file
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
        self._other_axes = tuple(
            axis for axis in self._tiff_axes if axis not in _tmp.axes.lower()
        )  # Axes that are not present in a single tiff page

        if "c" in self._tiff_axes:
            self._register_get_frame(self.get_frame_2D, "cyx")
        else:
            self._register_get_frame(self.get_frame_2D, "yx")

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        _tiff = tifffile.TiffFile(self.path).series[0]

        out_shape = tuple(self.sizes[axis] for axis in self.bundle_axes)
        out = np.zeros(out_shape, dtype=self._dtype)

        # Axes that are present in the tiff page
        page_axes = tuple(
            axis for axis in self._tiff_axes if axis not in self._other_axes
        )
        # Axes that need to be broadcasted from page to output
        broadcast_axes = tuple(
            axis
            for axis in self._tiff_axes
            if axis in self.bundle_axes and axis not in self._other_axes
        )

        page_indices = product(
            *[
                range(self.sizes[axis])
                if axis in self.bundle_axes
                else range(ind[axis], ind[axis] + 1)
                for axis in self._other_axes
            ]
        )

        # We iterate over all tiff pages to find the pages that are relevant for this frame
        for page_ind in page_indices:
            this_ind = {axis: index for axis, index in zip(self._other_axes, page_ind)}

            i = 0
            for j, axis in enumerate(self._other_axes):
                i += this_ind[axis] * np.prod(
                    [self.sizes[axis] for axis in self._other_axes[j + 1 :]],
                    dtype=int,
                )

            # Prepare selectors
            page_selector_list: list[Union[slice, int]] = []
            for axis in page_axes:
                if axis in self.bundle_axes:
                    page_selector_list.append(slice(None))
                else:
                    page_selector_list.append(ind[axis])
            page_selector = tuple(page_selector_list)

            out_selector_list: list[Union[slice, int]] = []
            for axis in self.bundle_axes:
                if axis in broadcast_axes:
                    out_selector_list.append(slice(None))  # broadcast
                else:
                    out_selector_list.append(
                        this_ind[axis]
                    )  # set page in a slice of the output
            out_selector = tuple(out_selector_list)

            page = _tiff.asarray(key=i)
            assert len(out_selector) == out.ndim
            assert len(page_selector) == page.ndim
            out[out_selector] = page[page_selector]

        return out

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tiff_shape

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

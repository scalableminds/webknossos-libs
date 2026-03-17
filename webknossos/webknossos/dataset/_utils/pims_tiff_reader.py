from itertools import product

import numpy as np
from pims import FramesSequenceND
from upath import UPath

try:
    import tifffile
except ImportError as e:
    raise ImportError(
        "Cannot import tifffile, please install it e.g. using 'webknossos[tifffile]'"
    ) from e


class PimsTiffReader(FramesSequenceND):
    @classmethod
    def class_exts(cls) -> set[str]:
        return {"tif", "tiff"}

    # class_priority is used in pims to pick the reader with the highest priority.
    # We decided to use a custom reader for tiff files to support images with more than 3 dimensions out of the box.
    # Default is 10, and bioformats priority is 2.
    # See http://soft-matter.github.io/pims/v0.6.1/custom_readers.html#plugging-into-pims-s-open-function
    class_priority = 19

    def __init__(self, path: UPath) -> None:
        super().__init__()

        self.path = UPath(path)

        with self.path.open("rb") as f:
            _tiff = tifffile.TiffFile(f).series[0]
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
            self._page_axes = tuple(
                axis for axis in self._tiff_axes if axis not in self._other_axes
            )

            if "c" in self._tiff_axes:
                self._register_get_frame(self.get_frame_2D, "cyx")
            else:
                self._register_get_frame(self.get_frame_2D, "yx")

    def get_frame_2D(self, **ind: int) -> np.ndarray:
        out_shape = tuple(self.sizes[axis] for axis in self.bundle_axes)
        out = np.zeros(out_shape, dtype=self._dtype)

        # Axes that are in bundle_axes AND require page selection (_other_axes).
        # These must be iterated so every page is written to the correct output slot.
        bundled_page_axes = [
            axis
            for axis in self._tiff_axes
            if axis in self.bundle_axes and axis in self._other_axes
        ]

        # Page axes not in bundle_axes: fixed by default_coords, indexed away after reading.
        extra_page_axes = [
            axis for axis in self._page_axes if axis not in self.bundle_axes
        ]

        with self.path.open("rb") as f:
            pages = tifffile.TiffFile(f).series[0].pages

            for bundled_page_coords in (
                product(*[range(self.sizes[axis]) for axis in bundled_page_axes])
                if bundled_page_axes
                else product()
            ):
                page_coords = dict(zip(bundled_page_axes, bundled_page_coords))
                current_ind = {**ind, **page_coords}

                # Compute flat page index from all page-selecting axes
                page_idx = (
                    int(
                        np.ravel_multi_index(
                            [current_ind[axis] for axis in self._other_axes],
                            [self.sizes[axis] for axis in self._other_axes],
                        )
                    )
                    if self._other_axes
                    else 0
                )

                page = pages[page_idx]
                assert page is not None, f"Page {page_idx} not found in TIFF file."
                page_data = page.asarray()

                # Index away page axes that are not part of bundle_axes (e.g. S in ZCYXS)
                if extra_page_axes:
                    page_data = page_data[
                        tuple(
                            current_ind[axis]
                            if axis in extra_page_axes
                            else slice(None)
                            for axis in self._page_axes
                        )
                    ]

                # Write to the correct position in out
                out[
                    tuple(
                        page_coords[axis] if axis in page_coords else slice(None)
                        for axis in self.bundle_axes
                    )
                ] = page_data

        return out

    @property
    def pixel_type(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tiff_shape

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return self._shape

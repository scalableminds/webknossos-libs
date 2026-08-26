from itertools import product

import numpy as np
from upath import UPath

from ...geometry.constants import C_AXIS, X_AXIS, Y_AXIS
from ...utils import WkImportError
from .image_source_registry import register_slice_reader
from .slice_reader import SliceReader

try:
    import tifffile
except ImportError as e:
    raise WkImportError("tifffile", "tifffile") from e


@register_slice_reader
class TiffSliceReader(SliceReader):
    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"tif", "tiff"}

    # The highest class_priority wins among eligible readers. 10 is the
    # default; this reader is preferred because it recognizes axis
    # information and supports tiffs with more than 3 dimensions.
    class_priority = 19

    def __init__(self, path: UPath) -> None:
        super().__init__()

        self.path = UPath(path)

        with self.path.open("rb") as f:
            _tiff = tifffile.TiffFile(f).series[0]
            raw_axes = tuple(_tiff.axes.lower())  # All the axes in the tiff file

            # Selecting the first page to get the dtype, shape and photometric tag
            if hasattr(_tiff, "pages"):
                _tmp = _tiff.pages[0]
            else:
                _tmp = _tiff["pages"][0]  # type: ignore
            assert _tmp is not None, "No pages found in tiff file."
            self._dtype = _tmp.dtype or np.dtype("uint8")
            self._shape = _tmp.shape

            # tifffile names the samples-per-pixel axis "S", only naming it "C"
            # when the file carries explicit multi-channel axis metadata. A
            # samples axis of 3 tagged as RGB is unambiguous colour, so it is
            # treated as the channel axis rather than as extra z-slices,
            # matching WEBKNOSSOS's own rule that only three uint8 channels
            # display as RGB (see channels_fit_one_layer). "S" and "C" can
            # both be present (e.g. ZCYXS); in that case "S" is left alone
            # and stays a plain extra axis.
            treat_s_as_c = (
                "s" in raw_axes
                and "c" not in raw_axes
                and _tiff.shape[raw_axes.index("s")] == 3
                and _tiff.keyframe.photometric == tifffile.PHOTOMETRIC.RGB
                and self._dtype.name == "uint8"
            )
            axis_rename = {"s": "c"} if treat_s_as_c else {}
            self.channels_are_rgb = treat_s_as_c

            self._tiff_axes = tuple(axis_rename.get(a, a) for a in raw_axes)
            for axis, shape in zip(self._tiff_axes, _tiff.shape):
                self._init_axis(axis, shape)

            page_axes = tuple(axis_rename.get(a, a) for a in _tmp.axes.lower())
            self._other_axes = tuple(
                axis for axis in self._tiff_axes if axis not in page_axes
            )  # Axes that are not present in a single tiff page
            self._page_axes = tuple(
                axis for axis in self._tiff_axes if axis not in self._other_axes
            )

            if C_AXIS in self._tiff_axes:
                self._set_get_slice(self.get_slice_2d, (C_AXIS, Y_AXIS, X_AXIS))
            else:
                self._set_get_slice(self.get_slice_2d, (Y_AXIS, X_AXIS))

    def get_slice_2d(self, **ind: int) -> np.ndarray:
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
            tiff = tifffile.TiffFile(f)
            series = tiff.series[0]

            # truncated tiff series (for example ImageJ virtual stacks) have only 1 real IFD
            # but store all frames contiguously at series.dataoffset. Reading them
            # via series.pages[i] fails for i > 0; use direct byte seeks instead.
            use_direct_seek = series.is_truncated and series.dataoffset is not None
            if use_direct_seek:
                frame_bytes = int(np.prod(self._shape)) * self._dtype.itemsize
                raw_dtype = np.dtype(tiff.byteorder + self._dtype.str[1:])
            else:
                pages = series.pages

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

                if use_direct_seek:
                    assert series.dataoffset is not None
                    f.seek(series.dataoffset + page_idx * frame_bytes)
                    raw = f.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        raise OSError(
                            f"Premature end of file while reading frame {page_idx}. "
                            f"Expected {frame_bytes} bytes, got {len(raw)}."
                        )
                    page_data: np.ndarray = (
                        np.frombuffer(raw, dtype=raw_dtype)
                        .reshape(self._shape)
                        .astype(self._dtype)
                    )
                else:
                    try:
                        page = pages[page_idx]
                    except IndexError:
                        raise ValueError(f"Page {page_idx} not found in TIFF file.")
                    if page is None:
                        raise ValueError(f"Page {page_idx} not found in TIFF file.")
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

                # page_data's remaining axes are in the page's on-disk order,
                # which does not always match bundle_axes (e.g. the samples
                # axis trails y/x on disk but leads as "c" in bundle_axes).
                remaining_page_axes = [
                    axis for axis in self._page_axes if axis not in extra_page_axes
                ]
                dest_axes = [
                    axis for axis in self.bundle_axes if axis not in page_coords
                ]
                if remaining_page_axes != dest_axes:
                    page_data = np.transpose(
                        page_data, [remaining_page_axes.index(a) for a in dest_axes]
                    )

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

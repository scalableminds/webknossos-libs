from typing import Any

import numpy as np
import pytest
from PIL import Image
from upath import UPath

from webknossos.dataset._image_conversion.common_slice_readers import (
    MultiImageSliceReader,
    SingleImageSliceReader,
)
from webknossos.dataset._image_conversion.image_source_registry import open_slice_reader
from webknossos.dataset._image_conversion.slice_reader import (
    SliceReader,
    _SlicedView,
)
from webknossos.dataset._image_conversion.tiff_slice_reader import TiffSliceReader
from webknossos.dataset.errors import (
    ImageConversionError,
    UnsupportedImageFormatError,
)

# A volume with distinct values everywhere, so a wrong axis order or a wrong
# index cannot accidentally produce the expected result.
_VOLUME = np.arange(2 * 3 * 4 * 5, dtype="uint16").reshape(2, 3, 4, 5)  # z, c, y, x


class _ZcyxReader(SliceReader):
    """Declares a single reader method returning all four axes at once."""

    def __init__(self) -> None:
        super().__init__()
        for axis, size in zip("zcyx", _VOLUME.shape):
            self._init_axis(axis, size)
        self._set_get_slice(self._get_slice, "zcyx")

    def _get_slice(self, **ind: int) -> np.ndarray:
        del ind
        return _VOLUME

    @property
    def pixel_type(self) -> Any:
        return _VOLUME.dtype


class _YxcReader(SliceReader):
    """Declares a 2D+channel method."""

    def __init__(self) -> None:
        super().__init__()
        self._init_axis("z", _VOLUME.shape[0])
        self._init_axis("y", _VOLUME.shape[2])
        self._init_axis("x", _VOLUME.shape[3])
        self._init_axis("c", _VOLUME.shape[1])
        self._set_get_slice(self._get_slice, "yxc")

    def _get_slice(self, **ind: int) -> np.ndarray:
        # (c, y, x) -> (y, x, c) for the requested z
        return np.moveaxis(_VOLUME[ind["z"]], 0, -1)

    @property
    def pixel_type(self) -> Any:
        return _VOLUME.dtype


def test_axis_bookkeeping() -> None:
    reader = _ZcyxReader()

    assert reader.axes == ["z", "c", "y", "x"]
    assert reader.sizes == {"z": 2, "c": 3, "y": 4, "x": 5}
    assert reader.ndim == 4
    assert reader.dtype == np.dtype("uint16")

    # Without iter axes the sequence has exactly one slice.
    assert len(reader) == 1

    reader.bundle_axes = ["c", "y", "x"]
    reader.iter_axes = ["z"]
    assert len(reader) == 2
    assert reader.slice_shape == (3, 4, 5)
    assert reader.shape == (2, 3, 4, 5)


def test_init_axis_rejects_duplicates() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="already exists"):
        reader._init_axis("z", 7)


def test_default_coords_rejects_unknown_axis() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="does not exist"):
        reader.default_coords["q"] = 0


def test_bundle_and_iter_axes_reject_unknown_axes() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="do not exist"):
        reader.bundle_axes = ["y", "q"]
    with pytest.raises(ValueError, match="do not exist"):
        reader.iter_axes = ["q"]


def test_bundle_axes_removes_axis_from_iter_axes() -> None:
    reader = _ZcyxReader()
    reader.iter_axes = ["z", "c"]
    reader.bundle_axes = ["c", "y", "x"]
    assert reader.iter_axes == ["z"]


def test_iter_axes_last_axis_varies_fastest() -> None:
    reader = _ZcyxReader()
    reader.bundle_axes = ["y", "x"]
    reader.iter_axes = ["z", "c"]

    assert len(reader) == 2 * 3
    # index i maps to (z, c) = (i // 3, i % 3)
    for i in range(6):
        np.testing.assert_array_equal(reader.get_slice(i), _VOLUME[i // 3, i % 3])


def test_adapter_drops_surplus_axes() -> None:
    # "zcyx" declared, "yx" requested: z and c must be indexed away using
    # the iter coordinate and default_coords respectively. This is the path
    # the DM3/DM4 readers rely on.
    reader = _ZcyxReader()
    reader.bundle_axes = ["y", "x"]
    reader.iter_axes = ["z"]
    reader.default_coords["c"] = 2

    for z in range(2):
        np.testing.assert_array_equal(reader.get_slice(z), _VOLUME[z, 2])


def test_adapter_transposes_to_requested_order() -> None:
    # "yxc" declared, "cyx" requested.
    reader = _YxcReader()
    reader.bundle_axes = ["c", "y", "x"]
    reader.iter_axes = ["z"]

    for z in range(2):
        np.testing.assert_array_equal(reader.get_slice(z), _VOLUME[z])


def test_adapter_rejects_axes_the_reader_cannot_produce() -> None:
    # "yxc" declared, but "z" is asked for as part of the slice. A reader
    # declares one method covering every axis it can produce, so this is a
    # reader bug rather than something to paper over by looping — refusing
    # keeps it from surfacing later as a silently mis-shaped array.
    reader = _YxcReader()
    with pytest.raises(ValueError, match=r"\['z'\] were requested"):
        reader.bundle_axes = ["z", "c", "y", "x"]


def test_get_slice_rejects_out_of_range_index() -> None:
    reader = _ZcyxReader()
    reader.bundle_axes = ["c", "y", "x"]
    reader.iter_axes = ["z"]
    with pytest.raises(IndexError):
        reader.get_slice(2)


class _RangeSequence(SliceReader):
    """A sequence whose slice i is a 1x1 array holding i."""

    def __init__(self, length: int = 6) -> None:
        super().__init__()
        self._init_axis("z", length)
        self._init_axis("y", 1)
        self._init_axis("x", 1)
        self._set_get_slice(self._read, "yx")
        self.iter_axes = ["z"]

    def _read(self, **coords: int) -> np.ndarray:
        return np.full((1, 1), coords["z"], dtype="uint8")

    @property
    def pixel_type(self) -> Any:
        return np.dtype("uint8")


def _values(view: Any) -> list[int]:
    return [int(np.asarray(s).ravel()[0]) for s in view]


def test_sequence_shape_and_iteration() -> None:
    seq = _RangeSequence()
    assert seq.shape == (6, 1, 1)
    assert _values(seq) == [0, 1, 2, 3, 4, 5]


def test_sliced_view_slicing_and_reversal() -> None:
    seq = _RangeSequence()

    assert isinstance(seq[1:4], _SlicedView)
    assert _values(seq[1:4]) == [1, 2, 3]
    assert _values(seq[::-1]) == [5, 4, 3, 2, 1, 0]
    assert len(seq[1:4]) == 3

    # Chained slicing, as copy_chunk_to_view does: select a range, reverse it, then
    # take a sub-range of that.
    assert _values(seq[1:5][::-1][0:2]) == [4, 3]


def test_sliced_view_indexing() -> None:
    seq = _RangeSequence()
    view = seq[2:5]

    assert int(view[0].ravel()[0]) == 2
    assert int(view[-1].ravel()[0]) == 4
    with pytest.raises(IndexError):
        view[3]


def test_sequence_negative_index() -> None:
    seq = _RangeSequence()
    assert int(seq[-1].ravel()[0]) == 5
    with pytest.raises(IndexError):
        seq[6]


def test_sliced_view_is_lazy() -> None:
    reads: list[int] = []

    class _CountingSequence(_RangeSequence):
        def get_slice(self, i: int) -> np.ndarray:
            reads.append(i)
            return super().get_slice(i)

    seq = _CountingSequence()
    view = seq[1:4][::-1]
    assert reads == [], "slicing must not read any slice"
    assert _values(view) == [3, 2, 1]
    assert reads == [3, 2, 1]


def _write_png(path: UPath, value: int) -> None:
    Image.fromarray(np.full((4, 6), value, dtype="uint8")).save(str(path))


def test_open_slice_reader_dispatches_on_extension(tmp_upath: UPath) -> None:
    png_path = tmp_upath / "single.png"
    _write_png(png_path, 7)
    assert isinstance(open_slice_reader(str(png_path)), SingleImageSliceReader)


def test_open_slice_reader_prefers_higher_class_priority(tmp_upath: UPath) -> None:
    # Both TiffSliceReader (priority=19) and, in principle, any lower-priority reader claim
    # .tif; the dedicated one has to win because it understands axis metadata.
    tif_path = tmp_upath / "single.tif"
    Image.fromarray(np.zeros((4, 6), dtype="uint8")).save(str(tif_path))
    assert isinstance(open_slice_reader(str(tif_path)), TiffSliceReader)


def test_open_slice_reader_rejects_unknown_extension(tmp_upath: UPath) -> None:
    unknown = tmp_upath / "data.unsupported"
    unknown.write_bytes(b"x")
    with pytest.raises(
        UnsupportedImageFormatError, match="Could not autodetect"
    ) as excinfo:
        open_slice_reader(str(unknown))

    # Dispatch failure raises the same public error as everything else in the
    # conversion path, so it is catchable as ImageConversionError (and as
    # ValueError), and it carries what it knows at this point. The caller
    # replaces it with a fuller one that also knows the path and the extras.
    error = excinfo.value
    assert isinstance(error, ImageConversionError)
    assert isinstance(error, ValueError)
    assert error.file_extension == "unsupported"
    assert "png" in error.supported_file_extensions


def test_open_slice_reader_rejects_extensionless_file(tmp_upath: UPath) -> None:
    extensionless = tmp_upath / "data"
    extensionless.write_bytes(b"x")
    with pytest.raises(UnsupportedImageFormatError, match="no extension"):
        open_slice_reader(str(extensionless))


def test_open_slice_reader_glob_with_multiple_matches(tmp_upath: UPath) -> None:
    for i in range(3):
        _write_png(tmp_upath / f"img_{i}.png", i)
    reader = open_slice_reader(str(tmp_upath / "img_*.png"))
    assert isinstance(reader, MultiImageSliceReader)
    assert len(reader) == 3


def test_image_sequence_orders_glob_naturally(tmp_upath: UPath) -> None:
    # Lexicographic ordering would put img_10 before img_2 and silently
    # scramble the z order of a converted stack.
    for i in [1, 2, 10]:
        _write_png(tmp_upath / f"img_{i}.png", i)

    reader = MultiImageSliceReader(str(tmp_upath / "img_*.png"))
    assert _values(reader) == [1, 2, 10]


def test_image_sequence_keeps_explicit_list_order(tmp_upath: UPath) -> None:
    # A caller-supplied list keeps its order — Dataset.from_images has already
    # applied z_slices_sort_key by the time the reader sees it.
    for i in [1, 2, 10]:
        _write_png(tmp_upath / f"img_{i}.png", i)

    paths = [str(tmp_upath / f"img_{i}.png") for i in [10, 1, 2]]
    assert _values(MultiImageSliceReader(paths)) == [10, 1, 2]


def test_image_sequence_reads_zip_archive(tmp_upath: UPath) -> None:
    import zipfile

    for i in [1, 2, 10]:
        _write_png(tmp_upath / f"img_{i}.png", i)
    archive = tmp_upath / "images.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        for i in [1, 2, 10]:
            zf.write(str(tmp_upath / f"img_{i}.png"), f"img_{i}.png")

    reader = MultiImageSliceReader(str(archive))
    assert reader.slice_shape == (4, 6)
    assert _values(reader) == [1, 2, 10]
    reader.close()


def test_image_sequence_reports_missing_files(tmp_upath: UPath) -> None:
    with pytest.raises(OSError, match="No files were found"):
        MultiImageSliceReader(str(tmp_upath / "nothing_*.png"))

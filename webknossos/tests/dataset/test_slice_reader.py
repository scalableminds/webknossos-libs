from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from PIL import Image
from upath import UPath

import webknossos as wk
from webknossos.dataset._image_conversion.common_slice_readers import (
    MultiImageSliceReader,
    SingleImageSliceReader,
)
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.image_source_registry import open_slice_reader
from webknossos.dataset._image_conversion.slice_reader import (
    SliceReader,
    _SlicedView,
)
from webknossos.dataset._image_conversion.sliced_image_source import SlicedImageSource
from webknossos.dataset._image_conversion.tiff_slice_reader import TiffSliceReader
from webknossos.dataset.errors import (
    ImageConversionError,
    UnsupportedImageFormatError,
)
from webknossos.geometry.constants import C_AXIS, X_AXIS, Y_AXIS, Z_AXIS
from webknossos.geometry.mag import Mag
from webknossos.geometry.normalized_bounding_box import NormalizedBoundingBox

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
        self._init_axis(Z_AXIS, _VOLUME.shape[0])
        self._init_axis(Y_AXIS, _VOLUME.shape[2])
        self._init_axis(X_AXIS, _VOLUME.shape[3])
        self._init_axis(C_AXIS, _VOLUME.shape[1])
        self._set_get_slice(self._get_slice, "yxc")

    def _get_slice(self, **ind: int) -> np.ndarray:
        # (c, y, x) -> (y, x, c) for the requested z
        return np.moveaxis(_VOLUME[ind[Z_AXIS]], 0, -1)

    @property
    def pixel_type(self) -> Any:
        return _VOLUME.dtype


def test_axis_bookkeeping() -> None:
    reader = _ZcyxReader()

    assert reader.axes == [Z_AXIS, C_AXIS, Y_AXIS, X_AXIS]
    assert reader.sizes == {Z_AXIS: 2, C_AXIS: 3, Y_AXIS: 4, X_AXIS: 5}
    assert reader.ndim == 4
    assert reader.dtype == np.dtype("uint16")

    # Without iter axes the sequence has exactly one slice.
    assert len(reader) == 1

    reader.bundle_axes = [C_AXIS, Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]
    assert len(reader) == 2
    assert reader.slice_shape == (3, 4, 5)
    assert reader.shape == (2, 3, 4, 5)


def test_init_axis_rejects_duplicates() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="already exists"):
        reader._init_axis(Z_AXIS, 7)


def test_default_coords_rejects_unknown_axis() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="does not exist"):
        reader.default_coords["q"] = 0


def test_bundle_and_iter_axes_reject_unknown_axes() -> None:
    reader = _ZcyxReader()
    with pytest.raises(ValueError, match="do not exist"):
        reader.bundle_axes = [Y_AXIS, "q"]
    with pytest.raises(ValueError, match="do not exist"):
        reader.iter_axes = ["q"]


def test_bundle_axes_removes_axis_from_iter_axes() -> None:
    reader = _ZcyxReader()
    reader.iter_axes = [Z_AXIS, C_AXIS]
    reader.bundle_axes = [C_AXIS, Y_AXIS, X_AXIS]
    assert reader.iter_axes == [Z_AXIS]


def test_iter_axes_last_axis_varies_fastest() -> None:
    reader = _ZcyxReader()
    reader.bundle_axes = [Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS, C_AXIS]

    assert len(reader) == 2 * 3
    # index i maps to (z, c) = (i // 3, i % 3)
    for i in range(6):
        np.testing.assert_array_equal(reader.get_slice(i), _VOLUME[i // 3, i % 3])


def test_adapter_drops_surplus_axes() -> None:
    # "zcyx" declared, "yx" requested: z and c must be indexed away using
    # the iter coordinate and default_coords respectively. This is the path
    # the DM3/DM4 readers rely on.
    reader = _ZcyxReader()
    reader.bundle_axes = [Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]
    reader.default_coords[C_AXIS] = 2

    for z in range(2):
        np.testing.assert_array_equal(reader.get_slice(z), _VOLUME[z, 2])


def test_adapter_transposes_to_requested_order() -> None:
    # "yxc" declared, "cyx" requested.
    reader = _YxcReader()
    reader.bundle_axes = [C_AXIS, Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]

    for z in range(2):
        np.testing.assert_array_equal(reader.get_slice(z), _VOLUME[z])


def test_adapter_rejects_axes_the_reader_cannot_produce() -> None:
    # "yxc" declared, but Z_AXIS is asked for as part of the slice. A reader
    # declares one method covering every axis it can produce, so this is a
    # reader bug rather than something to paper over by looping — refusing
    # keeps it from surfacing later as a silently mis-shaped array.
    reader = _YxcReader()
    with pytest.raises(ValueError, match=r"\['z'\] were requested"):
        reader.bundle_axes = [Z_AXIS, C_AXIS, Y_AXIS, X_AXIS]


def test_get_slice_rejects_out_of_range_index() -> None:
    reader = _ZcyxReader()
    reader.bundle_axes = [C_AXIS, Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]
    with pytest.raises(IndexError):
        reader.get_slice(2)


class _RangeSequence(SliceReader):
    """A sequence whose slice i is a 1x1 array holding i."""

    def __init__(self, length: int = 6) -> None:
        super().__init__()
        self._init_axis(Z_AXIS, length)
        self._init_axis(Y_AXIS, 1)
        self._init_axis(X_AXIS, 1)
        self._set_get_slice(self._read, "yx")
        self.iter_axes = [Z_AXIS]

    def _read(self, **coords: int) -> np.ndarray:
        return np.full((1, 1), coords[Z_AXIS], dtype="uint8")

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


# t, s, y, x — two non-"c" axes that are stepped through slice by slice
# (iterated, as opposed to "y"/"x", which are bundled into each slice), and
# no "z" axis at all.
_NO_Z_VOLUME = np.arange(2 * 3 * 4 * 5, dtype="uint16").reshape(2, 3, 4, 5)


class _TsyxReader(SliceReader):
    """Declares "t" and "s" as iterated axes; has no "z" axis."""

    def __init__(self) -> None:
        super().__init__()
        for axis, size in zip("tsyx", _NO_Z_VOLUME.shape):
            self._init_axis(axis, size)
        self._set_get_slice(self._get_slice, "tsyx")

    def _get_slice(self, **ind: int) -> np.ndarray:
        del ind
        return _NO_Z_VOLUME

    @property
    def pixel_type(self) -> Any:
        return _NO_Z_VOLUME.dtype


class _NoZImageSource(SlicedImageSource):
    """A SlicedImageSource backed by `_TsyxReader`, bypassing real file I/O."""

    def __init__(self) -> None:
        super().__init__(UPath("synthetic"), ReadOptions())

    @contextmanager
    def _open_slice_reader(self) -> Generator[SliceReader]:
        # Mirrors the base implementation's post-discovery setup (see
        # SlicedImageSource._open_slice_reader), which this override skips by
        # not going through the real file-opening machinery.
        images = _TsyxReader()
        if hasattr(self, "_bundle_axes"):
            images.default_coords.update(self._default_coords)
            images.bundle_axes = self._bundle_axes
            images.iter_axes = self._iter_axes
        yield images


def test_no_real_z_axis_gets_a_size_one_axis_without_relabeling_a_real_axis(
    tmp_upath: UPath,
) -> None:
    # The pipeline always wants explicit x, y and z axes (plus "c" for
    # channels); every other axis ("t", "s", ...) is iterated/split into
    # separate outputs instead. Some formats step through 2+ axes besides
    # "c" (here "t" and "s") but have no genuine "z" axis. expected_bbox must
    # add a size-1 "z" axis rather than relabeling a real axis (which would
    # corrupt its meaning — a real time axis reported as depth), and
    # copy_chunk_to_view must still read/write every (t, s) slice correctly
    # despite "z" not corresponding to any real iteration.
    source = _NoZImageSource()
    box = source.expected_bbox

    assert Z_AXIS in box.axes
    assert box.size.z == 1
    assert box.size.c == 1
    assert box.size[box.axes.index("t")] == 2  # from _TsyxReader / _NO_Z_VOLUME
    assert box.size[box.axes.index("s")] == 3

    ds = wk.Dataset(tmp_upath / "ds", voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "test",
        category="color",
        dtype=source.dtype,
        num_channels=1,
        data_format="zarr3",
    )
    layer.bounding_box = source.initial_layer_bounding_box(box)
    # tiny chunk/shard shape to reduce IO in tests
    mag_view = layer.add_mag(1, compress=False, chunk_shape=8, shard_shape=8)

    # Capture the output without going through any actual storage layer or doing file IO
    written: list[tuple[NormalizedBoundingBox, np.ndarray]] = []
    real_write = mag_view._array.write

    def _capture_write(bbox: NormalizedBoundingBox, data: np.ndarray) -> None:
        written.append((bbox, data.copy()))
        real_write(bbox, data)

    mag_view._array.write = _capture_write  # type: ignore[method-assign]

    axes = layer.normalized_bounding_box.axes
    t_index, s_index = axes.index("t"), axes.index("s")
    chunks = source.chunk_grid(
        layer.normalized_bounding_box, mag_view=mag_view, mag=Mag(1), batch_size=None
    )
    for chunk_bbox in chunks:
        source.copy_chunk_to_view(chunk_bbox, mag_view=mag_view, dtype=None)

    assert len(written) == 2 * 3  # one chunk per (t, s) combination

    for chunk_bbox, data in written:
        t = chunk_bbox.topleft[t_index]
        s = chunk_bbox.topleft[s_index]
        written_slice = data.reshape(4, 5)  # squeeze the size-1 s/t/c/z axes
        np.testing.assert_array_equal(written_slice, _NO_Z_VOLUME[t, s])

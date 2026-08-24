import importlib
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from shutil import copy
from tempfile import TemporaryDirectory, mkdtemp
from time import gmtime, strftime
from typing import Any

import h5py
import mrcfile
import numpy as np
import pytest
import tensorstore as ts
from cluster_tools import SequentialExecutor, get_executor
from numpy.typing import DTypeLike
from PIL import Image
from tifffile import TiffFile, imwrite
from upath import UPath

import webknossos as wk
from tests.constants import TESTDATA_DIR
from tests.data_fixtures import (
    create_synthetic_czi,
    create_synthetic_multi_timepoint_ims,
    download_wklibs_sample_archive,
    write_n5_array,
    write_neuroglancer_precomputed_scale,
    write_ome_zarr_v3_group,
    write_ozx_file,
    write_zarr_v3_array,
)
from tests.utils import HAS_PYLIBCZIRW, PYLIBCZIRW_EXPECTED, requires_pylibczirw


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_compare_tifffile(tmp_upath: UPath) -> None:
    ds = wk.Dataset(tmp_upath, (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            "testdata/tiff/test.02*.tiff",
            layer_name="compare_tifffile",
            compress=True,
            category="segmentation",
            topleft=(100, 100, 55),
            chunk_shape=(8, 8, 8),
            shard_shape=(64, 64, 64),
            executor=executor,
        )
    assert layer.bounding_box.topleft == wk.Vec3Int(100, 100, 55)
    data = layer.get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with TiffFile("testdata/tiff/test.0200.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        np.testing.assert_array_equal(data[:, :, z_index], comparison_slice)


# Testing different path types in addition to the mrc conversion
@pytest.mark.parametrize("path_type", [str, Path, UPath])
def test_mrc_from_images(tmp_upath: UPath, path_type: Callable[[UPath], Any]) -> None:
    Z, Y, X = 6, 24, 32
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            path_type(mrc_path),
            layer_name="mrc_layer",
            executor=executor,
        )

    assert layer.dtype == np.dtype("uint16")
    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]  # drop channel dim
    # Dataset stores as (x, y, z); original data is (z, y, x) → transpose
    np.testing.assert_array_equal(read_data, data.transpose(2, 1, 0))


@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("flip_z", [False, True])
@pytest.mark.parametrize("swap_xy", [False, True])
# Every flip mirrors the whole source extent, so a shard_shape smaller than the
# image is what distinguishes that from mirroring within each chunk. With the
# 64-cube the data fits one shard and the two are indistinguishable.
@pytest.mark.parametrize("shard_shape", [(64, 64, 64), (16, 16, 8)])
def test_mrc_from_images_flip_and_swap(
    tmp_upath: UPath,
    shard_shape: tuple[int, int, int],
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    swap_xy: bool,
) -> None:
    Z, Y, X = 6, 24, 32
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            mrc_path,
            layer_name="mrc_layer",
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            swap_xy=swap_xy,
            chunk_shape=(8, 8, 8),
            shard_shape=shard_shape,
            executor=executor,
        )
    actual = layer.get_finest_mag().read()[0]

    # flips apply in source axis order (z, y, x): flip_z mirrors z, flip_x
    # mirrors y and flip_y mirrors x (the SlicedImageSource convention).
    expected = data
    if flip_z:
        expected = expected[::-1]
    if flip_x:
        expected = expected[:, ::-1]
    if flip_y:
        expected = expected[:, :, ::-1]
    expected = expected.transpose(1, 2, 0) if swap_xy else expected.transpose(2, 1, 0)
    np.testing.assert_array_equal(actual, expected)


# layer.bounding_box is in Mag(1) but shard_shape is in mag space, so mag > 1
# only exercises the conversion between them once the image spans several
# shards; with one shard the mismatch cancels out.
@pytest.mark.parametrize("mag", [1, 2, 4])
@pytest.mark.parametrize("shard_shape", [(16, 16, 8), (64, 64, 64)])
def test_mrc_from_images_mag(
    tmp_upath: UPath, mag: int, shard_shape: tuple[int, int, int]
) -> None:
    Z, Y, X = 6, 24, 32
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            mrc_path,
            layer_name="mrc_layer",
            mag=mag,
            chunk_shape=(8, 8, 8),
            shard_shape=shard_shape,
            executor=executor,
        )

    # The layer bbox is Mag(1), so it scales with mag; the data stored at that
    # mag is the source unchanged.
    assert layer.bounding_box.size.to_tuple() == (X * mag, Y * mag, Z * mag)
    np.testing.assert_array_equal(
        layer.get_finest_mag().read()[0], data.transpose(2, 1, 0)
    )


def test_mrc_from_images_mag_parallel_compressed(tmp_upath: UPath) -> None:
    # Each job must own a whole shard. If chunks come out smaller than a shard
    # — as they do when a Mag(1) bbox is chunked with a mag-space shard_shape —
    # several workers write into the same compressed shard concurrently and
    # corrupt it. That is invisible under SequentialExecutor, so this needs a
    # real parallel executor to catch.
    Z, Y, X = 8, 64, 96
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with get_executor("multiprocessing", max_workers=4) as executor:
        layer = ds.add_layer_from_images(
            mrc_path,
            layer_name="mrc_layer",
            mag=2,
            compress=True,
            chunk_shape=(16, 16, 8),
            shard_shape=(32, 32, 8),
            executor=executor,
        )

    np.testing.assert_array_equal(
        layer.get_finest_mag().read()[0], data.transpose(2, 1, 0)
    )


def test_mrc_from_images_multi_shard_bbox(tmp_upath: UPath) -> None:
    # With an explicit shard_shape smaller than the image extent, conversion
    # must split into multiple shards along x and y. The final bounding box
    # must reflect the full image extent regardless.
    Z, Y, X = 6, 24, 32
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            mrc_path,
            layer_name="mrc_layer",
            chunk_shape=(8, 8, 8),
            shard_shape=(16, 16, 8),
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]
    np.testing.assert_array_equal(read_data, data.transpose(2, 1, 0))


def _read_ims_reference(ims_path: UPath, channel: int) -> np.ndarray:
    # Read independently via h5py/imaris_ims_file_reader rather than through
    # ImsImageSource, to get a reference unrelated to the code under test.
    from imaris_ims_file_reader.ims import ims as ImsFile

    ims_obj = ImsFile(str(ims_path), squeeze_output=False)
    _, _, z, y, x = ims_obj.shape
    ims_obj.close()
    with h5py.File(str(ims_path), "r") as hf:
        data = np.array(
            hf[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        )
    # This particular fixture's DataSetInfo metadata (used by ims_reader.shape)
    # declares smaller extents than the raw stored array, so crop to match.
    return data[:z, :y, :x]  # (z, y, x)


@pytest.mark.parametrize("channel", [0, 1])
def test_ims_from_images(tmp_upath: UPath, channel: int) -> None:
    ims_path = download_wklibs_sample_archive("brain_crop3.ims")

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path,
            layer_name="ims_layer",
            channel=channel,
            executor=executor,
        )

    assert layer.dtype == np.dtype("uint16")
    assert layer.num_channels == 1
    read_data = layer.get_finest_mag().read()[0]  # drop channel dim, shape (x, y, z)
    reference = _read_ims_reference(ims_path, channel)  # (z, y, x)
    np.testing.assert_array_equal(read_data, reference.transpose(2, 1, 0))


def test_ims_from_images_multi_shard_bbox(tmp_upath: UPath) -> None:
    # With an explicit shard_shape smaller than the image extent, conversion
    # must split into multiple shards along x and y. The final bounding box
    # must reflect the *full* image extent, not just a single shard's size —
    # a per-chunk-shape-based correction (as used for the SlicedImageSource path)
    # would be wrong here, since each ChunkedImageSource job only reports its own
    # shard-sized chunk, not the total extent.
    ims_path = download_wklibs_sample_archive("brain_crop3.ims")

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path,
            layer_name="ims_layer",
            channel=0,
            chunk_shape=(32, 32, 32),
            shard_shape=(256, 256, 32),
            executor=executor,
        )

    read_data = layer.get_finest_mag().read()[0]
    reference = _read_ims_reference(ims_path, 0)
    expected = reference.transpose(2, 1, 0)
    assert layer.bounding_box.size.to_tuple() == expected.shape
    np.testing.assert_array_equal(read_data, expected)


# The fixture is 673x635x51, so with shard shape (256, 256, 32) it spans several shards in every
# axis while with (1024, 1024, 64) it fits in one. Each flip mirrors the whole source
# extent, which only differs from mirroring within each chunk once the image
# spans more than one shard.
@pytest.mark.parametrize("shard_shape", [(1024, 1024, 64), (256, 256, 32)])
def test_ims_from_images_flip_and_swap(
    tmp_upath: UPath, shard_shape: tuple[int, int, int]
) -> None:
    # .ims files are read exclusively through ImsImageSource (never through
    # SlicedImageSource), so there's no separate "slow path" to compare against.
    # Instead,
    # this derives the expected flip/swap transform directly from the h5py
    # reference: flip_z/flip_x/flip_y reverse the source's z/y/x axes
    # respectively (in that source-axis order, regardless of swap_xy), and
    # swap_xy then picks (y, x, z) instead of (x, y, z) as the output order.
    ims_path = download_wklibs_sample_archive("brain_crop3.ims")

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path,
            layer_name="flipped",
            channel=0,
            flip_x=True,
            flip_y=True,
            flip_z=True,
            swap_xy=True,
            chunk_shape=(32, 32, 32),
            shard_shape=shard_shape,
            executor=executor,
        )
    actual = layer.get_finest_mag().read()[0]

    reference = _read_ims_reference(ims_path, 0)  # (z, y, x)
    intermediate = reference[::-1, ::-1, ::-1]  # flip_z, flip_x, flip_y
    expected = intermediate.transpose(1, 2, 0)  # swap_xy -> (y, x, z)

    np.testing.assert_array_equal(actual, expected)


def test_timepoint_argument_is_removed(tmp_upath: UPath) -> None:
    # All timepoints land in one layer on a "t" axis, so there is nothing left
    # to select; the argument is gone rather than silently ignored.
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with pytest.raises(TypeError, match="timepoint"):
        ds.add_layer_from_images(
            tmp_upath / "unused.ims",
            layer_name="x",
            timepoint=0,  # type: ignore[call-arg]
        )


def test_add_layer_from_images_unsupported_format(tmp_upath: UPath) -> None:
    # No reader handles .dcm. Downstream code needs to recognize this without
    # matching on an error message.
    unsupported = tmp_upath / "scan.dcm"
    unsupported.write_bytes(b"\x00" * 132)
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(wk.UnsupportedImageFormatError) as excinfo:
        ds.add_layer_from_images(unsupported, layer_name="color")
    error = excinfo.value
    assert error.file_extension == "dcm"
    assert error.path == unsupported
    assert error.missing_extras == ()
    assert "dcm" not in error.supported_file_extensions
    # Subclassing ValueError keeps `except ValueError` callers working.
    assert isinstance(error, ValueError)


@pytest.mark.parametrize(
    "filename,contents,wraps_reader_error",
    [
        ("broken.tif", b"not a tiff at all", True),
        # mrcfile parses this permissively and reports an empty extent instead
        # of raising, so there is no reader error to wrap — and without the
        # check the file would convert into an empty layer.
        ("broken.mrc", b"\x00" * 2048, False),
        # An HDF5 signature with nothing behind it: the .ims reader gets far
        # enough to fail deep inside h5py.
        ("broken.ims", b"\x89HDF\r\n\x1a\n" + b"\x00" * 100, True),
    ],
)
def test_add_layer_from_images_corrupt_file(
    tmp_upath: UPath, filename: str, contents: bytes, wraps_reader_error: bool
) -> None:
    # A damaged file of a *supported* format needs a different message than an
    # unsupported format.
    corrupt = tmp_upath / filename
    corrupt.write_bytes(contents)
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(wk.CorruptImageError) as excinfo:
        ds.add_layer_from_images(corrupt, layer_name="color")
    error = excinfo.value
    assert error.path == corrupt
    assert not isinstance(error, wk.UnsupportedImageFormatError)
    if wraps_reader_error:
        # The reader's own error stays available for the log.
        assert error.__cause__ is not None


def test_ims_empty_shape_is_corrupt(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A damaged or incompletely uploaded .ims file can still open and report a
    # shape, just with an empty axis (mirroring broken.mrc above), which would
    # otherwise convert into an empty layer instead of surfacing as corrupt.
    ims_path = tmp_upath / "empty.ims"
    ims_path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
    ims_image_source = importlib.import_module(
        "webknossos.dataset._image_conversion.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((1, 1, 4, 8, 0), np.dtype("uint16")),
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(wk.CorruptImageError) as excinfo:
        ds.add_layer_from_images(ims_path, layer_name="color")
    assert excinfo.value.path == ims_path


def test_add_layer_from_images_missing_file_is_not_corrupt(tmp_upath: UPath) -> None:
    # open_slice_reader flattens each reader's exception into a message, so a missing
    # file reaches the same code path as a damaged one. Telling the user their
    # file is damaged when it is simply not there would be worse than the
    # unspecific error they get today.
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(ValueError) as excinfo:
        ds.add_layer_from_images(tmp_upath / "does_not_exist.tif", layer_name="color")
    assert not isinstance(excinfo.value, wk.ImageConversionError)


def test_add_layer_from_images_rejects_unstorable_dtype(tmp_upath: UPath) -> None:
    # Float images cannot become a segmentation layer. The dtype comes from the
    # image, so this is an input problem the user can act on.
    float_image = tmp_upath / "float.tif"
    imwrite(str(float_image), np.zeros((8, 8), dtype="float32"))
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(wk.UnsupportedImageDataError, match="float32") as excinfo:
        ds.add_layer_from_images(float_image, layer_name="seg", category="segmentation")
    assert isinstance(excinfo.value, ValueError)


def test_add_layer_from_images_names_missing_optional_dependency(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With webknossos[ims] uninstalled, ImsImageSource never registers and the
    # file falls through to SlicedImageSource, which has no reader for it either. The error
    # must still point at the missing extra instead of calling .ims unsupported.
    # The test env installs every extra, so unregister the reader to reproduce
    # exactly the state a missing dependency leaves behind.
    registry = importlib.import_module(
        "webknossos.dataset._image_conversion.image_source_registry"
    )
    monkeypatch.setattr(registry, "_CHUNKED_IMAGE_SOURCE_CLASSES", [])
    monkeypatch.setattr(registry, "_UNAVAILABLE_EXTENSIONS", {"ims": "ims"})
    ims_path = tmp_upath / "a.ims"
    ims_path.write_bytes(b"stand-in for an ims file")
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with pytest.raises(
        wk.UnsupportedImageFormatError, match=r"pip install webknossos\[ims\]"
    ) as excinfo:
        ds.add_layer_from_images(ims_path, layer_name="color")
    assert excinfo.value.missing_extras == ("ims",)


def test_ims_from_images_multi_timepoint(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Multi-timepoint .ims files (without an explicit `timepoint=`) are a new
    # capability of the ChunkedImageSource abstraction: the bounding box gets a "t"
    # axis, and each chunk along it reads its own timepoint.
    ims_path = tmp_upath / "synthetic_multi_t.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path, num_timepoints=3, num_channels=1, z=4, y=8, x=10
    )
    ims_image_source = importlib.import_module(
        "webknossos.dataset._image_conversion.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((3, 1, 4, 8, 10), np.dtype("uint16")),
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path,
            layer_name="multi_t",
            data_format="zarr3",
            executor=executor,
        )

    assert layer.bounding_box.axes == ("t", "x", "y", "z")
    assert layer.bounding_box.size.to_tuple() == (3, 10, 8, 4)
    data = layer.get_finest_mag().read()  # (t, x, y, z), no channel dim
    for t in range(3):
        assert (data[t] == t * 100).all()


@pytest.mark.parametrize(
    "num_channels,dtype",
    [
        (2, np.uint8),
        (2, np.uint16),
        (3, np.uint16),
        (4, np.uint8),
        (4, np.uint16),
    ],
)
def test_ims_multi_channel_needs_one_layer_per_channel(
    tmp_upath: UPath,
    monkeypatch: pytest.MonkeyPatch,
    num_channels: int,
    dtype: DTypeLike,
) -> None:
    # .ims stores one acquisition channel per channel, so they belong in
    # separate layers whatever their number and dtype — except three uint8
    # channels, which fall back to a single RGB layer just like any other
    # format does without allow_multiple_layers=True (see
    # test_ims_three_uint8_channels_become_rgb_layer_without_allow_multiple_layers
    # below). Four channels are never treated as RGBA outside of an RGB image
    # format, regardless of dtype.
    ims_path = tmp_upath / "synthetic_multi_c.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path,
        num_timepoints=1,
        num_channels=num_channels,
        z=4,
        y=8,
        x=10,
        dtype=dtype,
    )
    ims_image_source = importlib.import_module(
        "webknossos.dataset._image_conversion.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((1, num_channels, 4, 8, 10), np.dtype(dtype)),
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        with pytest.raises(wk.UnsupportedImageDataError, match="allow_multiple_layers"):
            ds.add_layer_from_images(
                ims_path, layer_name="multi_c", data_format="zarr3", executor=executor
            )

        first_layer = ds.add_layer_from_images(
            ims_path,
            layer_name="multi_c",
            data_format="zarr3",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {
        f"multi_c__channel{c}" for c in range(num_channels)
    }
    assert first_layer.name == "multi_c__channel0"
    for c in range(num_channels):
        layer = ds.layers[f"multi_c__channel{c}"]
        assert layer.num_channels == 1
        assert layer.dtype == np.dtype(dtype)
        assert layer.bounding_box.axes == ("x", "y", "z")
        assert (layer.get_finest_mag().read()[0] == c).all()


def test_ims_three_uint8_channels_become_rgb_layer_without_allow_multiple_layers(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # .ims is not an RGB image format, but three uint8 channels still display
    # as RGB in WEBKNOSSOS, so without allow_multiple_layers=True they fall
    # back to a single layer instead of raising. Passing
    # allow_multiple_layers=True explicitly still splits them into one layer
    # per channel, same as any other channel count.
    num_channels = 3
    dtype = np.uint8
    ims_path = tmp_upath / "synthetic_multi_c.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path,
        num_timepoints=1,
        num_channels=num_channels,
        z=4,
        y=8,
        x=10,
        dtype=dtype,
    )
    ims_image_source = importlib.import_module(
        "webknossos.dataset._image_conversion.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((1, num_channels, 4, 8, 10), np.dtype(dtype)),
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path, layer_name="multi_c", data_format="zarr3", executor=executor
        )

        assert set(ds.layers.keys()) == {"multi_c"}
        assert layer.num_channels == num_channels
        assert layer.dtype == np.dtype(dtype)

        ds.add_layer_from_images(
            ims_path,
            layer_name="multi_c_split",
            data_format="zarr3",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"multi_c"} | {
        f"multi_c_split__channel{c}" for c in range(num_channels)
    }
    for c in range(num_channels):
        split_layer = ds.layers[f"multi_c_split__channel{c}"]
        assert split_layer.num_channels == 1
        assert split_layer.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "mode,expected_size",
    [("RGB", wk.VecInt(c=3, x=16, y=8, z=1)), ("RGBA", wk.VecInt(c=3, x=16, y=8, z=1))],
)
def test_rgb_image_stays_a_single_layer(
    tmp_upath: UPath, mode: str, expected_size: wk.VecInt
) -> None:
    # The RGB channels of an everyday image format belong together in one
    # layer, and an alpha channel is dropped rather than becoming a layer of
    # its own.
    png_path = tmp_upath / f"{mode.lower()}.png"
    data = np.zeros((8, 16, len(mode)), dtype="uint8")
    for channel in range(len(mode)):
        data[..., channel] = channel + 1
    Image.fromarray(data, mode=mode).save(str(png_path))
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            png_path, layer_name="color", executor=executor
        )

    assert set(ds.layers.keys()) == {"color"}
    assert layer.num_channels == 3
    assert layer.normalized_bounding_box.size == expected_size
    read = layer.get_finest_mag().read()
    for channel in range(3):
        assert (read[channel] == channel + 1).all()


def test_multi_channel_16bit_tiff_needs_one_layer_per_channel(
    tmp_upath: UPath,
) -> None:
    # .tif is not in the RGB-extension allowlist — TIFF can store RGB, but in
    # the microscopy context this library targets it usually doesn't — so even
    # three channels are separate acquisitions rather than RGB. uint16 makes
    # this unambiguous either way: only uint8 channels are ever treated as RGB.
    tiff_path = tmp_upath / "three_channels.tif"
    imwrite(
        str(tiff_path),
        np.zeros((3, 8, 16), dtype="uint16"),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        with pytest.raises(wk.UnsupportedImageDataError, match="allow_multiple_layers"):
            ds.add_layer_from_images(tiff_path, layer_name="color", executor=executor)

        ds.add_layer_from_images(
            tiff_path,
            layer_name="color",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {f"color__channel{c}" for c in range(3)}
    # Splitting a multi-channel image into layers defaults their colors to
    # red, green, blue, so the layers overlay sensibly right away.
    for c, expected_color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        view_configuration = ds.layers[f"color__channel{c}"].default_view_configuration
        assert view_configuration is not None
        assert view_configuration.color == expected_color


def test_multi_channel_tiff_layer_split_colors_beyond_rgb(
    tmp_upath: UPath,
) -> None:
    # A fourth (and any further) split-off channel layer still gets a color
    # of its own — just not one of the fixed RGB ones, since there is no
    # similarly universal convention past three channels.
    tiff_path = tmp_upath / "four_channels.tif"
    imwrite(
        str(tiff_path),
        np.zeros((4, 8, 16), dtype="uint16"),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        ds.add_layer_from_images(
            tiff_path,
            layer_name="color",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {f"color__channel{c}" for c in range(4)}
    colors = []
    for c in range(4):
        view_configuration = ds.layers[f"color__channel{c}"].default_view_configuration
        assert view_configuration is not None
        assert view_configuration.color is not None
        colors.append(view_configuration.color)

    assert colors[:3] == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # The fourth channel gets some other color, distinct from the first three.
    assert colors[3] not in colors[:3]
    assert all(0 <= component <= 255 for component in colors[3])


def test_three_uint8_channel_tiff_becomes_rgb_layer_without_allow_multiple_layers(
    tmp_upath: UPath,
) -> None:
    # .tif is not in the RGB-extension allowlist, but three uint8 channels
    # still display as RGB in WEBKNOSSOS, so without allow_multiple_layers=True
    # they fall back to a single layer instead of raising, same as the uint16
    # case above raises. allow_multiple_layers=True still splits them.
    tiff_path = tmp_upath / "three_channels.tif"
    imwrite(
        str(tiff_path),
        np.zeros((3, 8, 16), dtype="uint8"),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            tiff_path, layer_name="color", executor=executor
        )

        assert set(ds.layers.keys()) == {"color"}
        assert layer.num_channels == 3
        assert layer.dtype == np.dtype("uint8")

        ds.add_layer_from_images(
            tiff_path,
            layer_name="color_split",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"color"} | {
        f"color_split__channel{c}" for c in range(3)
    }


def test_add_layer_from_images_channel_argument_avoids_the_split(
    tmp_upath: UPath,
) -> None:
    # Picking one channel is the other way out of the error above, and has to
    # keep working without allow_multiple_layers.
    tiff_path = tmp_upath / "three_channels.tif"
    imwrite(
        str(tiff_path),
        np.zeros((3, 8, 16), dtype="uint16"),
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))

    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            tiff_path, layer_name="color", channel=1, executor=executor
        )

    assert set(ds.layers.keys()) == {"color"}
    assert layer.num_channels == 1


def test_ims_from_images_multi_timepoint_multi_channel_creates_multiple_layers(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With allow_multiple_layers=True, a multi-channel + multi-timepoint .ims
    # file should split into one layer per channel, each still keeping its
    # own "t" axis for all timepoints, rather than raising.
    ims_path = tmp_upath / "synthetic_multi_t_multi_c.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path, num_timepoints=2, num_channels=3, z=4, y=8, x=10
    )
    ims_image_source = importlib.import_module(
        "webknossos.dataset._image_conversion.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((2, 3, 4, 8, 10), np.dtype("uint16")),
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        ds.add_layer_from_images(
            ims_path,
            layer_name="multi_t_multi_c",
            data_format="zarr3",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {
        "multi_t_multi_c__channel0",
        "multi_t_multi_c__channel1",
        "multi_t_multi_c__channel2",
    }
    for c in range(3):
        layer = ds.layers[f"multi_t_multi_c__channel{c}"]
        assert layer.bounding_box.axes == ("t", "x", "y", "z")
        assert layer.bounding_box.size.to_tuple() == (2, 10, 8, 4)
        data = layer.get_finest_mag().read()  # (t, x, y, z), no channel dim
        for t in range(2):
            assert (data[t] == t * 100 + c).all()


# A single shard hides both the multi-shard flip bug and the mag/shard
# conversion, so every combination is checked at two shard shapes.
@requires_pylibczirw
@pytest.mark.parametrize("shard_shape", [(32, 32, 32), (64, 64, 64)])
@pytest.mark.parametrize("swap_xy", [False, True])
@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("flip_z", [False, True])
def test_czi_from_images_flip_and_swap(
    tmp_upath: UPath,
    shard_shape: tuple[int, int, int],
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
) -> None:
    # x != y != z so a transposed axis cannot pass by coincidence.
    czi_path = tmp_upath / "flip.czi"
    data = create_synthetic_czi(czi_path, z=6, y=40, x=48)[0, 0]  # (z, y, x)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            czi_path,
            layer_name="czi_layer",
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            swap_xy=swap_xy,
            data_format="zarr3",
            chunk_shape=(8, 8, 8),
            shard_shape=shard_shape,
            executor=executor,
        )
    actual = layer.get_finest_mag().read()[0]

    # flips apply in source axis order (z, y, x): flip_z mirrors z, flip_x
    # mirrors y and flip_y mirrors x (the ImageSource convention).
    expected = data
    if flip_z:
        expected = expected[::-1]
    if flip_x:
        expected = expected[:, ::-1]
    if flip_y:
        expected = expected[:, :, ::-1]
    expected = expected.transpose(1, 2, 0) if swap_xy else expected.transpose(2, 1, 0)
    np.testing.assert_array_equal(actual, expected)


@requires_pylibczirw
def test_czi_from_images_multi_timepoint(tmp_upath: UPath) -> None:
    # Timepoints land on a "t" axis within one layer rather than being pinned.
    czi_path = tmp_upath / "multi_t.czi"
    data = create_synthetic_czi(czi_path, num_timepoints=3, z=2, y=8, x=10)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            czi_path,
            layer_name="czi_layer",
            data_format="zarr3",
            executor=executor,
        )

    assert layer.bounding_box.axes == ("t", "x", "y", "z")
    assert layer.bounding_box.size.to_tuple() == (3, 10, 8, 2)
    actual = layer.get_finest_mag().read()
    for t in range(3):
        np.testing.assert_array_equal(actual[t], data[t, 0].transpose(2, 1, 0))


@requires_pylibczirw
def test_czi_from_images_splits_czi_channels_into_layers(tmp_upath: UPath) -> None:
    # A CZI "C" is a separate acquisition, not a RGB channel, so each one
    # becomes its own layer — and each must carry its own data.
    czi_path = tmp_upath / "multi_c.czi"
    data = create_synthetic_czi(czi_path, num_czi_channels=3, z=2, y=8, x=10)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        ds.add_layer_from_images(
            czi_path,
            layer_name="czi_layer",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers) == {
        "czi_layer__czi_channel0",
        "czi_layer__czi_channel1",
        "czi_layer__czi_channel2",
    }
    for c in range(3):
        layer = ds.layers[f"czi_layer__czi_channel{c}"]
        assert layer.num_channels == 1
        np.testing.assert_array_equal(
            layer.get_finest_mag().read()[0], data[0, c].transpose(2, 1, 0)
        )


@requires_pylibczirw
def test_czi_from_images_selects_a_single_czi_channel(tmp_upath: UPath) -> None:
    czi_path = tmp_upath / "pick_c.czi"
    data = create_synthetic_czi(czi_path, num_czi_channels=3, z=2, y=8, x=10)

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            czi_path, layer_name="czi_layer", czi_channel=2, executor=executor
        )

    assert len(ds.layers) == 1
    np.testing.assert_array_equal(
        layer.get_finest_mag().read()[0], data[0, 2].transpose(2, 1, 0)
    )


def test_compare_nd_tifffile(tmp_upath: UPath) -> None:
    four_d_series_tif = (
        download_wklibs_sample_archive("4D") / "4D_series" / "4D-series.ome.tif"
    )
    ds = wk.Dataset(tmp_upath, (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            str(four_d_series_tif),
            layer_name="color",
            category="color",
            topleft=(2, 55, 100, 100),
            data_format="zarr3",
            chunk_shape=(8, 8, 8),
            shard_shape=(64, 64, 64),
            executor=executor,
        )
    assert layer.bounding_box.topleft == wk.VecInt(t=2, z=55, y=100, x=100)
    assert layer.bounding_box.size == wk.VecInt(t=7, z=5, y=167, x=439)
    read_with_tifffile_reader = TiffFile(str(four_d_series_tif)).asarray()
    # For ND data without explicit channel axis, read() returns data directly
    # without a channel wrapper dimension
    read_from_dataset = layer.get_finest_mag().read()
    np.testing.assert_array_equal(read_with_tifffile_reader, read_from_dataset)


def _remote_repo_image_path(archive: str, *parts: str) -> Callable[[], str]:
    """Builds a lazy resolver for a file/dir/glob inside a wklibs-samples
    archive: the archive is only downloaded (once per process) when the
    returned callable is actually invoked, not when this is called."""

    def resolve() -> str:
        base = download_wklibs_sample_archive(archive)
        return str(base.joinpath(*parts)) if parts else str(base)

    return resolve


def _remote_single_multipage_tiff_folder() -> str:
    """`various_tiff_formats/test_C.tif` copied into its own folder, to test
    converting a folder that contains a single multi-page tiff."""
    various_tiff_formats_dir = download_wklibs_sample_archive("various_tiff_formats")
    folder = UPath(mkdtemp(prefix="single_multipage_tiff_folder-"))
    copy(str(various_tiff_formats_dir / "test_C.tif"), str(folder / "test_C.tif"))
    return str(folder)


REPO_IMAGES_ARGS: list[
    tuple[
        str | list[UPath] | Callable[[], str | list[UPath]],
        dict[str, Any],
        str,
        int,
        int,
        wk.VecInt,
    ]
] = [
    (
        "testdata/tiff/test.*.tiff",
        {"category": "segmentation"},
        "uint8",
        1,
        1,
        wk.VecInt(c=1, x=265, y=265, z=257),
    ),
    (
        [
            TESTDATA_DIR / "tiff" / "test.0000.tiff",
            TESTDATA_DIR / "tiff" / "test.0001.tiff",
            TESTDATA_DIR / "tiff" / "test.0002.tiff",
        ],
        {},
        "uint8",
        1,
        1,
        wk.VecInt(c=1, x=265, y=265, z=3),
    ),
    (
        "testdata/rgb_tiff/test_rgb.tif",
        {"mag": 2},
        "uint8",
        1,
        1,
        wk.VecInt(c=1, x=64, y=64, z=6),
    ),
    (
        "testdata/rgb_tiff",
        {"mag": 2, "channel": 0, "dtype": "uint32"},
        "uint32",
        1,
        1,
        wk.VecInt(c=1, x=64, y=64, z=6),
    ),
    (
        _remote_repo_image_path("temca2", "*", "*", "*.jpg"),
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        1,
        1,
        wk.VecInt(c=1, x=1024, y=1024, z=12),
    ),
    (
        _remote_repo_image_path("temca2"),
        {"flip_z": True, "batch_size": 2048},
        "uint8",
        1,
        1,
        # The topmost folder contains an extra image,
        # which is included here as well, but not in
        # the glob pattern above. Therefore z is +1.
        wk.VecInt(c=1, x=1024, y=1024, z=13),
    ),
    (
        _remote_repo_image_path("tiff_with_different_shapes", "*"),
        {"flip_y": True},
        "uint8",
        1,
        1,
        wk.VecInt(c=1, x=2970, y=2521, z=4),
    ),
    (
        _remote_repo_image_path("various_tiff_formats", "test_CS.tif"),
        {"data_format": "zarr3", "allow_multiple_layers": True},
        "uint8",
        1,
        5,
        wk.VecInt(s=3, x=64, c=1, y=128, z=128),
    ),
    (
        _remote_repo_image_path("various_tiff_formats", "test_C.tif"),
        {"allow_multiple_layers": True},
        "uint8",
        1,
        5,
        wk.VecInt(c=1, x=128, y=128, z=64),
    ),
    # same as test_C.tif above, but as a single file in a folder:
    (
        _remote_single_multipage_tiff_folder,
        {"allow_multiple_layers": True},
        "uint8",
        1,
        5,
        wk.VecInt(c=1, x=128, y=128, z=64),
    ),
    (
        _remote_repo_image_path("various_tiff_formats", "test_I.tif"),
        {},
        "uint32",
        1,
        1,
        wk.VecInt(c=1, x=64, y=128, z=64),
    ),
    (
        _remote_repo_image_path("various_tiff_formats", "test_S.tif"),
        {"data_format": "zarr3"},
        "uint16",
        1,
        1,
        wk.VecInt(s=3, x=64, y=128, z=128),
    ),
    (
        _remote_repo_image_path("4D", "single_channel", "single-channel.ome.tiff"),
        {},
        "int8",
        1,
        1,
        wk.VecInt(c=1, x=439, y=167, z=1),
    ),
    (
        _remote_repo_image_path(
            "4D", "multi_channel_z_series", "multi-channel-z-series.ome.tif"
        ),
        {"allow_multiple_layers": True},
        "int8",
        1,
        3,
        wk.VecInt(c=1, x=439, y=167, z=5),
    ),
]


def _test_repo_images(
    tmp_upath: UPath,
    path: str | list[UPath] | Callable[[], str | list[UPath]],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    num_layers: int,
    size: wk.VecInt,
) -> wk.Dataset:
    if callable(path):
        path = path()
    with SequentialExecutor() as executor:
        ds = wk.Dataset(tmp_upath, (1, 1, 1))
        layer = ds.add_layer_from_images(
            path,
            layer_name="color",
            compress=True,
            executor=executor,
            **kwargs,
        )
        assert layer.dtype == np.dtype(dtype)
        assert layer.num_channels == num_channels
        assert len(ds.layers) == num_layers
        assert layer.normalized_bounding_box.size == size
        if isinstance(layer, wk.SegmentationLayer):
            assert layer.largest_segment_id is not None
            assert layer.largest_segment_id > 0
    return ds


@pytest.mark.parametrize(
    "path, kwargs, dtype, num_channels, num_layers, size", REPO_IMAGES_ARGS
)
def test_repo_images(
    tmp_upath: UPath,
    path: str | list[UPath] | Callable[[], str | list[UPath]],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    num_layers: int,
    size: wk.VecInt,
) -> None:
    _test_repo_images(tmp_upath, path, kwargs, dtype, num_channels, num_layers, size)


def _remote_sample(name: str) -> Callable[[], UPath]:
    """Lazily downloads the named wklibs-samples archive (only when the
    returned callable is invoked, not at parametrize-collection time)."""
    return lambda: download_wklibs_sample_archive(name)


def _remote_samples(names: list[str]) -> Callable[[], list[UPath]]:
    return lambda: [download_wklibs_sample_archive(n) for n in names]


# All scif images used here are published with CC0 license,
# see https://scif.io/images.
TEST_IMAGES_ARGS: list[
    tuple[
        str,
        Callable[[], UPath | list[UPath]],
        dict,
        str,
        int,
        tuple[int, int, int],
    ]
] = [
    (
        "slice_0420.dm4",
        _remote_sample("slice_0420.dm4"),
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (8192, 8192, 1),
    ),
    (
        "slice_0073.dm3",
        _remote_sample("slice_0073.dm3"),
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (4096, 4096, 1),
    ),
    (
        "slice_0073.dm3...",
        _remote_samples(["slice_0073.dm3", "slice_0074.dm3"]),
        {"data_format": "zarr3"},  # using zarr to allow smaller chunking
        "uint16",
        1,
        (4096, 4096, 2),
    ),
    (
        "dnasample1.dm3",
        _remote_sample("dnasample1.dm3"),
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "int16",
        1,
        (4096, 4096, 1),
    ),
    (
        "scifio-test.gif",
        _remote_sample("scifio-test.gif"),
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "scifio-test.jpg",
        _remote_sample("scifio-test.jpg"),
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "scifio-test.png",
        _remote_sample("scifio-test.png"),
        {"flip_y": True},
        "uint8",
        3,
        (500, 500, 1),
    ),
]
if HAS_PYLIBCZIRW:
    TEST_IMAGES_ARGS.append(
        (
            # published with CC0 license, taken from
            # https://doi.org/10.6084/m9.figshare.c.3727411_D391.v1
            "embedded_NCI_mono_matrigelcollagen_docetaxel_day10_sample10.czi",
            _remote_sample(
                "embedded_NCI_mono_matrigelcollagen_docetaxel_day10_sample10.czi"
            ),
            {},
            "uint16",
            1,
            (512, 512, 30),
        )
    )
elif PYLIBCZIRW_EXPECTED:
    # pylibCZIrw is expected on this Python version; missing here means a
    # broken test environment, not a reason to silently drop CZI coverage.
    raise ImportError(
        "pylibCZIrw is not installed, but is expected for this Python version."
    )


def _test_test_images(
    tmp_upath: UPath,
    name: str,
    path: Callable[[], UPath | list[UPath]],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    size: tuple[int, int, int],
) -> wk.Dataset:
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        l_normal = ds.add_layer_from_images(
            path(),
            layer_name="normal_" + name,
            compress=True,
            executor=executor,
            **kwargs,
        )
        assert l_normal.dtype == np.dtype(dtype)
        assert l_normal.num_channels == num_channels
        assert l_normal.bounding_box.size.to_tuple() == size
    return ds


def test_zarr_array_from_images(tmp_upath: UPath) -> None:
    Z, Y, X = 6, 24, 32
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    zarr_path = tmp_upath / "test.zarr"
    write_zarr_v3_array(zarr_path, data, dimension_names=["z", "y", "x"])

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            zarr_path,
            layer_name="zarr_layer",
            executor=executor,
        )

    assert layer.dtype == np.dtype("uint16")
    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]  # drop channel dim
    np.testing.assert_array_equal(read_data, data.transpose(2, 1, 0))

    layer.downsample()
    assert len(layer.mags) > 1


def test_ome_zarr_from_images_picks_finest_resolution(tmp_upath: UPath) -> None:
    Z, Y, X = 4, 16, 16
    finest = np.arange(Z * Y * X, dtype="uint8").reshape(Z, Y, X)
    coarse = np.zeros((Z, Y // 2, X // 2), dtype="uint8")
    group_path = tmp_upath / "test.ome.zarr"
    axes = [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    write_ome_zarr_v3_group(
        group_path,
        [
            ("1", coarse, [1.0, 2.0, 2.0]),  # listed first, but coarser
            ("0", finest, [1.0, 1.0, 1.0]),
        ],
        axes,
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            group_path,
            layer_name="ome_zarr_layer",
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]
    np.testing.assert_array_equal(read_data, finest.transpose(2, 1, 0))


def test_ome_zarr_from_images_scale_option_picks_specified_level(
    tmp_upath: UPath,
) -> None:
    Z, Y, X = 4, 16, 16
    finest = np.arange(Z * Y * X, dtype="uint8").reshape(Z, Y, X)
    coarse = np.zeros((Z, Y // 2, X // 2), dtype="uint8")
    group_path = tmp_upath / "test.ome.zarr"
    axes = [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    write_ome_zarr_v3_group(
        group_path,
        [("0", finest, [1.0, 1.0, 1.0]), ("1", coarse, [1.0, 2.0, 2.0])],
        axes,
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            group_path,
            layer_name="ome_zarr_layer",
            scale=1,
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X // 2, Y // 2, Z)
    read_data = layer.get_finest_mag().read()[0]
    np.testing.assert_array_equal(read_data, coarse.transpose(2, 1, 0))


def test_ome_zarr_from_images_allow_multiple_layers_never_splits_by_scale(
    tmp_upath: UPath,
) -> None:
    # A multiscale, multi-channel source: allow_multiple_layers=True must
    # split by channel only, never additionally by resolution level — every
    # split-off layer converts a single (the finest) level.
    Z, Y, X = 2, 8, 8
    finest = np.arange(4 * Z * Y * X, dtype="uint8").reshape(4, Z, Y, X)
    coarse = np.zeros((4, Z, Y // 2, X // 2), dtype="uint8")
    group_path = tmp_upath / "test.ome.zarr"
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    write_ome_zarr_v3_group(
        group_path,
        [
            ("0", finest, [1.0, 1.0, 1.0, 1.0]),
            ("1", coarse, [1.0, 1.0, 2.0, 2.0]),
        ],
        axes,
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        ds.add_layer_from_images(
            group_path,
            layer_name="test",
            allow_multiple_layers=True,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {
        "test__channel0",
        "test__channel1",
        "test__channel2",
        "test__channel3",
    }
    for i in range(4):
        layer = ds.layers[f"test__channel{i}"]
        assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
        read_data = layer.get_finest_mag().read()[0]
        np.testing.assert_array_equal(read_data, finest[i].transpose(2, 1, 0))


def test_ozx_from_images_picks_finest_resolution(tmp_upath: UPath) -> None:
    Z, Y, X = 4, 16, 16
    finest = np.arange(Z * Y * X, dtype="uint8").reshape(Z, Y, X)
    coarse = np.zeros((Z, Y // 2, X // 2), dtype="uint8")
    ozx_path = tmp_upath / "test.ozx"
    axes = [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    write_ozx_file(
        ozx_path,
        [
            ("1", coarse, [1.0, 2.0, 2.0]),  # listed first, but coarser
            ("0", finest, [1.0, 1.0, 1.0]),
        ],
        axes,
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ozx_path,
            layer_name="ozx_layer",
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]
    np.testing.assert_array_equal(read_data, finest.transpose(2, 1, 0))


def test_ome_zarr_omero_channel_metadata_applied_when_splitting(
    tmp_upath: UPath,
) -> None:
    Z, Y, X = 2, 8, 8
    data = np.zeros((4, Z, Y, X), dtype="uint16")
    group_path = tmp_upath / "test.ome.zarr"
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    omero = {
        "channels": [
            {
                "color": "0000FF",
                "window": {"min": 0.0, "max": 65535.0, "start": 10.0, "end": 500.0},
                "label": "DAPI",
                "active": True,
            },
            {
                "window": {"min": 0.0, "max": 65535.0, "start": 0.0, "end": 100.0},
                "active": False,
            },
            {
                "color": "FF0000",
                "window": {"start": 5.0, "end": 200.0},
                "label": "Hyb probe!",
            },
            # Channel 3 has no matching omero entry at all.
        ]
    }
    write_ome_zarr_v3_group(
        group_path, [("0", data, [1.0, 1.0, 1.0, 1.0])], axes, omero
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        ds.add_layer_from_images(
            group_path,
            layer_name="test",
            allow_multiple_layers=True,
            executor=executor,
        )

    # Channels with a usable omero label are named from it (with disallowed
    # characters stripped); channels without one fall back to "channel{N}".
    assert set(ds.layers.keys()) == {
        "test__DAPI",
        "test__channel1",
        "test__Hybprobe",
        "test__channel3",
    }

    dapi = ds.layers["test__DAPI"].default_view_configuration
    assert dapi is not None
    assert dapi.color == (0, 0, 255)
    assert dapi.intensity_range == (10.0, 500.0)
    assert dapi.min == 0.0
    assert dapi.max == 65535.0
    assert dapi.is_disabled is False

    inactive = ds.layers["test__channel1"].default_view_configuration
    assert inactive is not None
    # No omero color for this channel, so it falls back to the usual
    # red/green/blue default for the second split-off layer.
    assert inactive.color == (0, 255, 0)
    assert inactive.intensity_range == (0.0, 100.0)
    assert inactive.is_disabled is True

    hyb = ds.layers["test__Hybprobe"].default_view_configuration
    assert hyb is not None
    assert hyb.color == (255, 0, 0)
    assert hyb.intensity_range == (5.0, 200.0)
    assert hyb.min is None
    assert hyb.max is None
    assert hyb.is_disabled is None

    fallback = ds.layers["test__channel3"].default_view_configuration
    assert fallback is not None
    fallback_color = fallback.color
    assert fallback_color is not None
    assert fallback_color not in {(0, 0, 255), (255, 0, 0)}
    assert all(0 <= component <= 255 for component in fallback_color)
    assert fallback.intensity_range is None


def test_ome_zarr_omero_channel_metadata_applied_to_pinned_channel(
    tmp_upath: UPath,
) -> None:
    Z, Y, X = 2, 8, 8
    data = np.zeros((2, Z, Y, X), dtype="uint16")
    group_path = tmp_upath / "test.ome.zarr"
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    omero = {
        "channels": [
            {"color": "00FF00", "window": {"start": 1.0, "end": 2.0}},
            {
                "color": "FF00FF",
                "window": {"start": 3.0, "end": 4.0},
                "label": "second",
            },
        ]
    }
    write_ome_zarr_v3_group(
        group_path, [("0", data, [1.0, 1.0, 1.0, 1.0])], axes, omero
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            group_path, layer_name="pinned", channel=1, executor=executor
        )

    # channel=<index> pins a single layer, so the omero label plays no part
    # in naming it — only the view configuration is applied.
    assert layer.name == "pinned"
    view_configuration = layer.default_view_configuration
    assert view_configuration is not None
    assert view_configuration.color == (255, 0, 255)
    assert view_configuration.intensity_range == (3.0, 4.0)


def test_n5_pyramid_from_images_picks_finest_level(tmp_upath: UPath) -> None:
    Z, Y, X = 4, 16, 16
    finest = np.arange(Z * Y * X, dtype="uint8").reshape(Z, Y, X)
    coarse = np.zeros((Z, Y // 2, X // 2), dtype="uint8")
    group_path = tmp_upath / "test.n5"
    write_n5_array(group_path / "s0", finest, downsampling_factors=[1, 1, 1])
    write_n5_array(group_path / "s1", coarse, downsampling_factors=[1, 2, 2])

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            group_path,
            layer_name="n5_layer",
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]
    np.testing.assert_array_equal(read_data, finest.transpose(2, 1, 0))


def test_neuroglancer_precomputed_from_images_picks_finest_scale(
    tmp_upath: UPath,
) -> None:
    X, Y, Z = 16, 16, 4
    finest = np.arange(X * Y * Z, dtype="uint8").reshape(X, Y, Z, 1)
    coarse = np.zeros((X // 2, Y // 2, Z, 1), dtype="uint8")
    volume_path = tmp_upath / "precomputed"
    write_neuroglancer_precomputed_scale(
        volume_path, finest, resolution=(4.0, 4.0, 4.0)
    )
    write_neuroglancer_precomputed_scale(
        volume_path, coarse, resolution=(8.0, 8.0, 4.0)
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            volume_path,
            layer_name="neuroglancer_layer",
            executor=executor,
        )

    assert layer.bounding_box.size.to_tuple() == (X, Y, Z)
    read_data = layer.get_finest_mag().read()[0]
    # Precomputed's native axis order is already (x, y, z) — same as wK storage.
    np.testing.assert_array_equal(read_data, finest[:, :, :, 0])


def test_real_ome_zarr_sample_conversion(tmp_upath: UPath) -> None:
    # A real published OME-Zarr v3 (NGFF 0.5) sample, not a synthetic
    # fixture: 5D (t, c, z, y, x), sharded, 3 resolution levels, with
    # dimension_names on both the group's OME axes and each array's own
    # zarr.json, plus a "consolidated_metadata" key this reader must ignore.
    sample_path = download_wklibs_sample_archive("13457537.zarr")

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            sample_path,
            layer_name="sample",
            channel=0,
            executor=executor,
        )

    assert layer.dtype == np.dtype("uint16")
    assert layer.bounding_box.axes == ("t", "x", "y", "z")
    assert layer.bounding_box.size.to_tuple() == (18, 198, 223, 12)

    # The sample's omero metadata for channel 0 ("cy 1"):
    # {"color": "FFFFFF", "window": {"min": 0.0, "max": 65535.0, "start": 0.0, "end": 1200.0}}
    view_configuration = layer.default_view_configuration
    assert view_configuration is not None
    assert view_configuration.color == (255, 255, 255)
    assert view_configuration.intensity_range == (0.0, 1200.0)
    assert view_configuration.min == 0.0
    assert view_configuration.max == 65535.0

    # Compares against the finest level's channel 0 read directly, rather
    # than trusting the converter's own read path.
    finest_array = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(sample_path / "0")},
        },
        open=True,
        context=ts.Context(),
    ).result()
    # (t, c, z, y, x), channel 0 -> (t, z, y, x) -> (t, x, y, z)
    expected = np.asarray(finest_array[:, 0].read().result()).transpose(0, 3, 2, 1)
    actual = layer.get_finest_mag().read()  # (t, x, y, z), channel dim dropped
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name, path, kwargs, dtype, num_channels, size", TEST_IMAGES_ARGS
)
@pytest.mark.skipif(
    "CI" not in os.environ or os.environ["CI"] != "true" or sys.platform != "linux",
    reason="only run on linux CI",
)
def test_test_images(
    tmp_upath: UPath,
    name: str,
    path: Callable[[], UPath | list[UPath]],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    size: tuple[int, int, int],
) -> None:
    _test_test_images(tmp_upath, name, path, kwargs, dtype, num_channels, size)


if __name__ == "__main__":
    time = lambda: strftime("%Y-%m-%d_%H-%M-%S", gmtime())  # noqa: E731

    for repo_image in REPO_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            image_path = repo_image[0]
            if callable(image_path):
                image_path = image_path()
            if isinstance(image_path, list):
                image_path = str(image_path[0])
            name = "".join(filter(str.isalnum, image_path))
            print(repo_image)
            print(
                _test_repo_images(UPath(tempdir), *repo_image)
                .upload(new_dataset_name=f"test_repo_images_{name}_{time()}")
                .url
            )

    for test_images_args in TEST_IMAGES_ARGS:
        with TemporaryDirectory() as tempdir:
            name = "".join(filter(str.isalnum, test_images_args[0]))
            print(*test_images_args)
            print(
                _test_test_images(UPath(tempdir), *test_images_args)
                .upload(new_dataset_name=f"test_test_images_{name}_{time()}")
                .url
            )

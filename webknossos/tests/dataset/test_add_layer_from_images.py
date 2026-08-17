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
from cluster_tools import SequentialExecutor, get_executor
from tifffile import TiffFile
from upath import UPath

import webknossos as wk
from tests.constants import TESTDATA_DIR
from tests.data_fixtures import (
    create_synthetic_multi_timepoint_ims,
    download_and_unpack,
    download_wklibs_sample_archive,
)


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
    # mirrors y and flip_y mirrors x (the PimsImages convention).
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
    # ImsChunkedImages, to get a reference unrelated to the code under test.
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
    # a per-chunk-shape-based correction (as used for the generic pims path)
    # would be wrong here, since each ChunkedImages job only reports its own
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
    # .ims files are read exclusively through ImsChunkedImages (never through
    # pims), so there's no separate "slow path" to compare against. Instead,
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


def test_ims_from_images_multi_timepoint(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Multi-timepoint .ims files (without an explicit `timepoint=`) are a new
    # capability of the ChunkedImages abstraction: the bounding box gets a "t"
    # axis, and each chunk along it reads its own timepoint.
    ims_path = tmp_upath / "synthetic_multi_t.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path, num_timepoints=3, num_channels=1, z=4, y=8, x=10
    )
    ims_chunked_images = importlib.import_module(
        "webknossos.dataset._utils.ims_chunked_images"
    )
    monkeypatch.setattr(
        ims_chunked_images,
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


def test_ims_from_images_multi_timepoint_multi_channel_single_layer_keeps_both_axes(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # expected_bbox reports every axis the source actually has, so a file with
    # both multiple timepoints and multiple (3+) channels fits in a single
    # layer as (t, c, x, y, z) — "c" is where NormalizedBoundingBox reads
    # num_channels from, and NDBoundingBox.chunk() keeps it whole. Splitting
    # channels into separate layers is what allow_multiple_layers=True does
    # instead — see test_ims_from_images_multi_timepoint_multi_channel_creates_multiple_layers.
    ims_path = tmp_upath / "synthetic_multi_t_multi_c.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path, num_timepoints=2, num_channels=3, z=4, y=8, x=10
    )
    ims_chunked_images = importlib.import_module(
        "webknossos.dataset._utils.ims_chunked_images"
    )
    monkeypatch.setattr(
        ims_chunked_images,
        "_read_ims_metadata_quietly",
        lambda _path: ((2, 3, 4, 8, 10), np.dtype("uint16")),
    )

    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        layer = ds.add_layer_from_images(
            ims_path,
            layer_name="multi_t_multi_c",
            data_format="zarr3",
            executor=executor,
        )

    assert layer.bounding_box.axes == ("t", "c", "x", "y", "z")
    assert layer.bounding_box.size.to_tuple() == (2, 3, 10, 8, 4)
    assert layer.num_channels == 3
    data = layer.get_finest_mag().read()
    assert data.shape == (2, 3, 10, 8, 4)
    for t in range(2):
        for c in range(3):
            assert (data[t, c] == t * 100 + c).all()


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
    ims_chunked_images = importlib.import_module(
        "webknossos.dataset._utils.ims_chunked_images"
    )
    monkeypatch.setattr(
        ims_chunked_images,
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


# All scif images used here are published with CC0 license,
# see https://scif.io/images.
TEST_IMAGES_ARGS: list[
    tuple[
        str | list[str],
        str | list[str],
        dict,
        str,
        int,
        tuple[int, int, int],
    ]
] = [
    (
        "https://static.webknossos.org/data/webknossos-libs/slice_0420.dm4",
        "slice_0420.dm4",
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (8192, 8192, 1),
    ),
    (
        "https://static.webknossos.org/data/webknossos-libs/slice_0073.dm3",
        "slice_0073.dm3",
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "uint16",
        1,
        (4096, 4096, 1),
    ),
    (
        [
            "https://static.webknossos.org/data/webknossos-libs/slice_0073.dm3",
            "https://static.webknossos.org/data/webknossos-libs/slice_0074.dm3",
        ],
        ["slice_0073.dm3", "slice_0074.dm3"],
        {"data_format": "zarr3"},  # using zarr to allow smaller chunking
        "uint16",
        1,
        (4096, 4096, 2),
    ),
    (
        "https://static.webknossos.org/data/wklibs-samples/dnasample1.zip",
        "dnasample1.dm3",
        {"data_format": "zarr3"},  # using zarr to allow z=1 chunking
        "int16",
        1,
        (4096, 4096, 1),
    ),
    (
        # published with CC0 license, taken from
        # https://doi.org/10.6084/m9.figshare.c.3727411_D391.v1
        "https://static.webknossos.org/data/wklibs-samples/embedded_NCI_mono_matrigelcollagen_docetaxel_day10_sample10.czi",
        "embedded_NCI_mono_matrigelcollagen_docetaxel_day10_sample10.czi",
        {},
        "uint16",
        1,
        (512, 512, 30),
    ),
    (
        "https://static.webknossos.org/data/wklibs-samples/test-gif.zip",
        "scifio-test.gif",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://static.webknossos.org/data/wklibs-samples/test-jpeg2000.zip",
        "scifio-test.jp2",
        {},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://static.webknossos.org/data/wklibs-samples/test-jpg.zip",
        "scifio-test.jpg",
        {"flip_x": True, "batch_size": 2048},
        "uint8",
        3,
        (500, 500, 1),
    ),
    (
        "https://static.webknossos.org/data/wklibs-samples/test-png.zip",
        "scifio-test.png",
        {"flip_y": True},
        "uint8",
        3,
        (500, 500, 1),
    ),
]


def _test_test_images(
    tmp_upath: UPath,
    url: str | list[str],
    filename: str | list[str],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    size: tuple[int, int, int],
) -> wk.Dataset:
    unzip_path = tmp_upath / "unzip"
    download_and_unpack(url, unzip_path, filename)
    path: UPath | list[UPath]
    if isinstance(filename, list):
        layer_name = filename[0] + "..."
        path = [unzip_path / i for i in filename]
    else:
        layer_name = filename
        path = unzip_path / filename
    ds = wk.Dataset(tmp_upath / "ds", (1, 1, 1))
    with SequentialExecutor() as executor:
        l_normal = ds.add_layer_from_images(
            path,
            layer_name="normal_" + layer_name,
            compress=True,
            executor=executor,
            **kwargs,
        )
        assert l_normal.dtype == np.dtype(dtype)
        assert l_normal.num_channels == num_channels
        assert l_normal.bounding_box.size.to_tuple() == size
    return ds


@pytest.mark.parametrize(
    "url, filename, kwargs, dtype, num_channels, size", TEST_IMAGES_ARGS
)
@pytest.mark.skipif(
    "CI" not in os.environ or os.environ["CI"] != "true" or sys.platform != "linux",
    reason="only run on linux CI",
)
def test_test_images(
    tmp_upath: UPath,
    url: str | list[str],
    filename: str | list[str],
    kwargs: dict,
    dtype: str,
    num_channels: int,
    size: tuple[int, int, int],
) -> None:
    _test_test_images(tmp_upath, url, filename, kwargs, dtype, num_channels, size)


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
            name = "".join(filter(str.isalnum, test_images_args[1]))
            print(*test_images_args)
            print(
                _test_test_images(UPath(tempdir), *test_images_args)
                .upload(new_dataset_name=f"test_test_images_{name}_{time()}")
                .url
            )

import importlib
import json
import warnings
from collections.abc import Iterator
from shutil import copytree
from typing import Any
from unittest.mock import MagicMock, patch

import mrcfile
import numpy as np
import pytest
from cluster_tools import SequentialExecutor
from PIL import Image
from tifffile import TiffFile, imwrite
from upath import UPath

from tests.constants import TESTDATA_DIR
from tests.utils import (
    create_synthetic_multi_timepoint_ims,
    create_synthetic_zarr3_array,
    download_ims_fixture,
)
from webknossos.dataset import (
    Dataset,
    RemoteDataset,
    UnsupportedImageFormatError,
)
from webknossos.dataset._utils.image_source import ReadOptions
from webknossos.dataset._utils.mrc_image_source import MrcImageSource
from webknossos.dataset._utils.tiff_slices import TiffSlices
from webknossos.geometry import BoundingBox, Vec3Int, VecInt


@pytest.fixture(autouse=True, scope="function")
def ignore_warnings() -> Iterator:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="webknossos", message=r"\[WARNING\]")
        yield


def test_compare_tifffile(tmp_upath: UPath) -> None:
    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            TESTDATA_DIR / "tiff",
            tmp_upath,
            (1, 1, 1),
            compress=True,
            layer_name="tiff_stack",
            layer_category="segmentation",
            shard_shape=(256, 256, 256),
            map_filepath_to_layer_name=Dataset.ConversionLayerMapping.ENFORCE_SINGLE_LAYER,
            executor=executor,
        )
    assert len(ds.layers) == 1
    assert "tiff_stack" in ds.layers
    data = ds.layers["tiff_stack"].get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with (
            (TESTDATA_DIR / "tiff" / "test.0000.tiff").open("rb") as f,
            TiffFile(f) as tif_file,
        ):
            comparison_slice = tif_file.asarray().T
        np.testing.assert_array_equal(data[:, :, z_index], comparison_slice)


def test_ZCYX_tiff(tmp_upath: UPath) -> None:
    # Y > X is required to expose the bug: with Y <= X the wrong indexing silently
    # broadcasts channel-0 data into all channels instead of raising an error.
    data = np.random.randint(0, 1000, (5, 4, 7, 6), dtype="uint16")
    tif_path = tmp_upath / "test_ZCYX.tif"
    imwrite(str(tif_path), data, imagej=True)
    assert TiffFile(str(tif_path)).series[0].axes == "ZCYX"
    assert TiffFile(str(tif_path)).series[0].shape == (5, 4, 7, 6)
    assert len(TiffFile(str(tif_path)).pages) == 5 * 4  # Z*C
    assert TiffFile(str(tif_path)).pages[0].axes == "YX"

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            tif_path,
            tmp_upath / "ds",
            (1, 1, 1),
            data_format="zarr3",
            executor=executor,
        )
    assert len(ds.layers) == 4
    assert ds.get_color_layers()[0].bounding_box.size == Vec3Int(x=6, y=7, z=5)


def test_imagej_virtual_stack_tiff(tmp_upath: UPath) -> None:
    # ImageJ virtual stacks store all frames contiguously after a single IFD
    # (series.is_truncated=True). series.pages only exposes that 1 IFD, so
    # reading any frame beyond index 0 used to raise IndexError.
    Z, Y, X = 8, 16, 12
    data = np.arange(Z * Y * X, dtype="uint16").reshape(Z, Y, X)
    tif_path = tmp_upath / "test_virtual_stack.tif"
    imwrite(str(tif_path), data, imagej=True, truncate=True, metadata={"axes": "ZYX"})

    t = TiffFile(str(tif_path))
    assert t.series[0].is_truncated, (
        "expected an ImageJ virtual stack (truncated series)"
    )
    assert len(t.pages) == 1, "expected exactly 1 real IFD"
    assert t.series[0].shape == (Z, Y, X)

    reader = TiffSlices(tif_path)
    reader.bundle_axes = ["y", "x"]
    reader.iter_axes = ["z"]

    assert reader.shape == (Z, Y, X)
    assert reader.slice_shape == (Y, X)

    for z in range(Z):
        np.testing.assert_array_equal(np.array(reader[z]), data[z])

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            tif_path,
            tmp_upath / "ds",
            (1, 1, 1),
            executor=executor,
        )
    assert ds.get_color_layers()[0].bounding_box.size == Vec3Int(x=X, y=Y, z=Z)
    result = ds.get_color_layers()[0].get_finest_mag().read()[0]
    np.testing.assert_array_equal(result, data.transpose(2, 1, 0))


def test_tiled_CZYX_tiff(tmp_upath: UPath) -> None:
    import tifffile as tifffile_module

    C, Z, Y, X = 3, 2, 32, 32
    tile = (16, 16)
    data = np.arange(C * Z * Y * X, dtype="uint16").reshape(C, Z, Y, X)
    tif_path = tmp_upath / "test_tiled_CZYX.tif"
    imwrite(str(tif_path), data, tile=tile, metadata={"axes": "CZYX"})

    assert TiffFile(str(tif_path)).series[0].axes == "CZYX"
    first_page = TiffFile(str(tif_path)).pages[0]
    assert isinstance(first_page, tifffile_module.TiffPage)
    assert first_page.is_tiled
    assert first_page.chunks == tile

    # Verify that reading z=0 only accesses the C pages for z=0, not pages from other z-slices.
    # With CZYX ordering (C=3, Z=2) pages are laid out as: c=0→[pg0,pg1], c=1→[pg2,pg3], c=2→[pg4,pg5]
    # so z=0 corresponds to pages 0, 2, 4 and z=1 to pages 1, 3, 5.
    reader = TiffSlices(tif_path)
    reader.bundle_axes = ["c", "y", "x"]
    reader.iter_axes = ["z"]

    pages_read: list[int] = []
    original_asarray = tifffile_module.TiffPage.asarray

    def tracking_asarray(self: tifffile_module.TiffPage, **kwargs: Any) -> np.ndarray:
        pages_read.append(self.index)
        return original_asarray(self, **kwargs)

    with patch.object(tifffile_module.TiffPage, "asarray", tracking_asarray):
        slice_z0 = np.array(reader[0])

    assert pages_read == [0, 2, 4], (
        f"Expected pages [0, 2, 4] for z=0, got {pages_read}"
    )
    assert slice_z0.shape == (C, Y, X)
    np.testing.assert_array_equal(slice_z0, data[:, 0, :, :])


def test_multiple_multitiffs(tmp_upath: UPath) -> None:
    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            TESTDATA_DIR / "various_tiff_formats",
            tmp_upath,
            (1, 1, 1),
            data_format="zarr3",
            layer_name="tiffs",
            executor=executor,
        )
    assert len(ds.layers) == 12

    expected_dtype_channels_size_per_layer = {
        "tiffs_test_CS.tif__channel0": (
            "uint8",
            1,
            VecInt(s=3, x=64, c=1, y=128, z=128),
        ),
        "tiffs_test_CS.tif__channel1": (
            "uint8",
            1,
            VecInt(s=3, x=64, c=1, y=128, z=128),
        ),
        "tiffs_test_CS.tif__channel2": (
            "uint8",
            1,
            VecInt(s=3, x=64, c=1, y=128, z=128),
        ),
        "tiffs_test_CS.tif__channel3": (
            "uint8",
            1,
            VecInt(s=3, x=64, c=1, y=128, z=128),
        ),
        "tiffs_test_CS.tif__channel4": (
            "uint8",
            1,
            VecInt(s=3, x=64, c=1, y=128, z=128),
        ),
        "tiffs_test_C.tif__channel0": ("uint8", 1, VecInt(c=1, x=128, y=128, z=64)),
        "tiffs_test_C.tif__channel1": ("uint8", 1, VecInt(c=1, x=128, y=128, z=64)),
        "tiffs_test_C.tif__channel2": ("uint8", 1, VecInt(c=1, x=128, y=128, z=64)),
        "tiffs_test_C.tif__channel3": ("uint8", 1, VecInt(c=1, x=128, y=128, z=64)),
        "tiffs_test_C.tif__channel4": ("uint8", 1, VecInt(c=1, x=128, y=128, z=64)),
        "tiffs_test_I.tif": ("uint32", 1, VecInt(c=1, x=64, y=128, z=64)),
        "tiffs_test_S.tif": ("uint16", 1, VecInt(s=3, z=64, x=128, y=128)),
    }

    for layer_name, layer in ds.layers.items():
        dtype, channels, size = expected_dtype_channels_size_per_layer[layer_name]
        assert layer.dtype == np.dtype(dtype)
        assert layer.num_channels == channels
        assert layer.normalized_bounding_box.size == size

        # Check that the zarr.json metadata is correct
        mag1 = layer.get_finest_mag()
        array_shape = json.loads((mag1.path / "zarr.json").read_bytes())["shape"]
        shard_aligned_bottomright = layer.normalized_bounding_box.with_bottomright_xyz(
            layer.bounding_box.bottomright_xyz.ceildiv(mag1.info.shard_shape)
            * mag1.info.shard_shape
        ).bottomright
        assert array_shape == shard_aligned_bottomright.to_list()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_rgb_image_creates_a_single_rgb_layer(tmp_upath: UPath, mode: str) -> None:
    # from_images() passes allow_multiple_layers=True, but the colour channels
    # of an everyday image format still belong in one layer rather than being
    # split into grayscale ones — and an alpha channel is dropped, not turned
    # into a fourth layer.
    images = tmp_upath / "images"
    images.mkdir()
    data = np.zeros((8, 16, len(mode)), dtype="uint8")
    for channel in range(len(mode)):
        data[..., channel] = channel + 1
    Image.fromarray(data, mode=mode).save(str(images / "shot.png"))

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            images, tmp_upath / "ds", (1, 1, 1), layer_name="color", executor=executor
        )

    assert set(ds.layers.keys()) == {"color"}
    layer = ds.get_layer("color")
    assert layer.num_channels == 3
    read = layer.get_finest_mag().read()
    for channel in range(3):
        assert (read[channel] == channel + 1).all()


def test_multi_channel_ims_creates_multiple_layers(tmp_upath: UPath) -> None:
    # brain_crop3.ims has 2 channels and no explicit channel is selected, so
    # ImsImageSource.get_possible_layers() reports {"channel": [0, 1]} and
    # from_images() (which always passes allow_multiple_layers=True) should
    # split it into one layer per channel instead of picking just the first.
    ims_path = download_ims_fixture(tmp_upath)

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            ims_path,
            tmp_upath / "ds",
            (1, 1, 1),
            layer_name="brain",
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"brain__channel0", "brain__channel1"}

    channel_data = {}
    for layer_name, layer in ds.layers.items():
        assert layer.dtype == np.dtype("uint16")
        assert layer.num_channels == 1
        assert layer.bounding_box.size.to_tuple() == (673, 635, 51)
        channel_data[layer_name] = layer.get_finest_mag().read()[0]

    # Sanity check the two layers actually hold different channel data,
    # rather than both being (accidentally) the same channel duplicated.
    assert not np.array_equal(
        channel_data["brain__channel0"], channel_data["brain__channel1"]
    )


def test_multi_channel_multi_timepoint_ims_creates_multiple_layers_with_t_axis(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # from_images() always passes allow_multiple_layers=True internally, so a
    # multi-channel + multi-timepoint .ims file should split into one layer
    # per channel (via get_possible_layers()'s {"channel": [...]} report),
    # each keeping its own "t" axis for all timepoints, rather than raising
    # the "multiple timepoints and multiple channels" ValueError that firing
    # on the unpinned from_images() discovery probe would otherwise cause.
    ims_path = tmp_upath / "synthetic_multi_t_multi_c.ims"
    create_synthetic_multi_timepoint_ims(
        ims_path, num_timepoints=2, num_channels=3, z=4, y=8, x=10
    )
    ims_image_source = importlib.import_module(
        "webknossos.dataset._utils.ims_image_source"
    )
    monkeypatch.setattr(
        ims_image_source,
        "_read_ims_metadata_quietly",
        lambda _path: ((2, 3, 4, 8, 10), np.dtype("uint16")),
    )

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            ims_path,
            tmp_upath / "ds",
            (1, 1, 1),
            layer_name="multi",
            executor=executor,
        )

    assert set(ds.layers.keys()) == {
        "multi__channel0",
        "multi__channel1",
        "multi__channel2",
    }
    for c in range(3):
        layer = ds.layers[f"multi__channel{c}"]
        assert layer.bounding_box.axes == ("t", "x", "y", "z")
        assert layer.bounding_box.size.to_tuple() == (2, 10, 8, 4)
        data = layer.get_finest_mag().read()  # (t, x, y, z), no channel dim
        for t in range(2):
            assert (data[t] == t * 100 + c).all()


def _open_mrc_chunked_images(mrc_path: UPath) -> MrcImageSource:
    return MrcImageSource(mrc_path, ReadOptions())


def test_mrc_chunked_images_metadata(tmp_upath: UPath) -> None:
    Z, Y, X = 5, 16, 32
    data = np.arange(Z * Y * X, dtype="float32").reshape(Z, Y, X)
    mrc_path = tmp_upath / "test.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    reader = _open_mrc_chunked_images(mrc_path)

    assert reader.dtype == np.dtype("float32")
    assert reader.num_channels == 1
    assert reader.channel is None
    assert reader.get_possible_layers() is None
    assert reader.expected_bbox.size.to_tuple() == (X, Y, Z)


def test_mrc_chunked_images_reopens_mmap_per_chunk(tmp_upath: UPath) -> None:
    # MRC data is read via a memory-mapped array; each copy_chunk_to_view() call must
    # reopen its own mmap (rather than reusing a shared/cached one), so no
    # mmap handle crosses a multiprocessing boundary between parallel jobs.
    Z, Y, X = 4, 8, 8
    data = np.zeros((Z, Y, X), dtype="uint16")
    mrc_path = tmp_upath / "test_reopen.mrc"
    with mrcfile.new(str(mrc_path), overwrite=True) as mrc:
        mrc.set_data(data)

    reader = _open_mrc_chunked_images(mrc_path)

    ds = Dataset(tmp_upath / "ds", (1, 1, 1))
    layer = ds.add_layer("mrc", category="color", dtype=reader.dtype)
    layer.bounding_box = reader.expected_bbox
    mag_view = layer.add_mag(1)

    open_count = 0
    original_mmap = mrcfile.mmap

    def counting_mmap(*args: Any, **kwargs: Any) -> Any:
        nonlocal open_count
        open_count += 1
        return original_mmap(*args, **kwargs)

    with patch("mrcfile.mmap", counting_mmap):
        for z in range(Z):
            bbox = BoundingBox((0, 0, z), (X, Y, 1))
            reader.copy_chunk_to_view(bbox, mag_view=mag_view, dtype=None)

    assert open_count == Z, (
        f"Expected mrcfile.mmap to be called {Z} times (once per chunk), got {open_count}"
    )


def test_no_slashes_in_layername(tmp_upath: UPath) -> None:
    (input_path := tmp_upath / "tiff" / "subfolder" / "tifffiles").mkdir(parents=True)
    copytree(
        str(TESTDATA_DIR / "tiff_with_different_shapes"),
        str(input_path),
        dirs_exist_ok=True,
    )

    for strategy in Dataset.ConversionLayerMapping:
        with SequentialExecutor() as executor:
            dataset = Dataset.from_images(
                tmp_upath / "tiff",
                tmp_upath / str(strategy),
                voxel_size=(10, 10, 10),
                map_filepath_to_layer_name=strategy,
                executor=executor,
            )

            assert all("/" not in layername for layername in dataset.layers)


def test_from_images_treats_zarr_stores_as_atomic_directories(
    tmp_upath: UPath,
) -> None:
    # A folder containing two zarr stores -- one named *.zarr, one bare --
    # plus an ordinary tiff must become three layers, without walking into
    # either store's internal chunk files.
    data_a = np.zeros((2, 5, 6), dtype=np.uint8)
    data_b = np.ones((2, 5, 6), dtype=np.uint8) * 7
    input_dir = tmp_upath / "input"
    create_synthetic_zarr3_array(
        input_dir / "layer_a.zarr", data_a, dimension_names=["z", "y", "x"]
    )
    create_synthetic_zarr3_array(
        input_dir / "layer_b", data_b, dimension_names=["z", "y", "x"]
    )
    imwrite(str(input_dir / "layer_c.tiff"), np.zeros((5, 6), dtype=np.uint8))

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            input_dir,
            tmp_upath / "ds",
            voxel_size=(1, 1, 1),
            # Mixing a 2D image with 3D stores at the same folder level is
            # ambiguous for the default INSPECT_SINGLE_FILE strategy (it
            # decides 2D-vs-3D handling for the whole folder from a single
            # representative file) -- orthogonal to what this test checks, so
            # sidestepped with an explicit one-layer-per-item strategy.
            map_filepath_to_layer_name=Dataset.ConversionLayerMapping.ENFORCE_LAYER_PER_FILE,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"layer_a.zarr", "layer_b", "layer_c.tiff"}
    np.testing.assert_array_equal(
        ds.layers["layer_b"].get_finest_mag().read()[0],
        data_b.transpose(2, 1, 0),
    )


def test_remote_dataset_from_images() -> None:
    """Test that RemoteDataset.from_images converts images and calls upload."""
    mock_remote_ds = MagicMock(spec=RemoteDataset)

    with patch.object(Dataset, "upload", return_value=mock_remote_ds) as mock_upload:
        with SequentialExecutor() as executor:
            result = RemoteDataset.from_images(
                TESTDATA_DIR / "tiff",
                voxel_size=(1, 1, 1),
                name="test_remote",
                compress=True,
                layer_name="tiff_layer",
                layer_category="segmentation",
                shard_shape=(256, 256, 256),
                map_filepath_to_layer_name=Dataset.ConversionLayerMapping.ENFORCE_SINGLE_LAYER,
                executor=executor,
                url="http://localhost:9000",
                token="test_token",
            )

    assert result is mock_remote_ds
    mock_upload.assert_called_once_with(
        new_dataset_name="test_remote",
        folder=None,
    )


def test_optional_reader_suffixes_match_class_exts() -> None:
    # _OPTIONAL_READERS has to restate each reader's suffixes, because a
    # reader whose dependency is missing never imports and so cannot report its
    # own class_exts(). Whenever a reader *is* importable, the two must agree —
    # otherwise the missing-dependency hint names the wrong formats, or
    # silently stops covering one. Covers both strategies: slice readers are
    # just as optional as chunked ones now that tifffile is not in the base
    # install.
    from webknossos.dataset._utils.chunked_image_source import ChunkedImageSource
    from webknossos.dataset._utils.image_source_registry import (
        _CHUNKED_READER_CLASSES,
        _OPTIONAL_READERS,
        _SLICE_READER_CLASSES,
        get_unavailable_suffixes,
    )
    from webknossos.dataset._utils.slice_sequence import SliceSequence

    # Annotated because the two lists' only common base is ABC, which does
    # not declare class_exts(); the union does.
    registered: dict[str, type[SliceSequence] | type[ChunkedImageSource]] = {
        cls.__name__: cls for cls in _SLICE_READER_CLASSES
    }
    registered.update({cls.__name__: cls for cls in _CHUNKED_READER_CLASSES})
    # The test env installs every extra except czi on Python 3.14, since
    # pylibCZIrw ships no wheel there.
    unavailable_extras = set(get_unavailable_suffixes().values())
    for reader in _OPTIONAL_READERS:
        if reader.extra in unavailable_extras:
            continue
        assert reader.class_name in registered, (
            f"{reader.class_name} is declared optional but did not register"
        )
        assert registered[reader.class_name].class_exts() == set(reader.suffixes), (
            f"declared suffixes for {reader.class_name} are out of sync with "
            "its class_exts()"
        )


def test_from_images_names_missing_optional_dependency(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With an extra uninstalled its reader never registers, so its formats are
    # simply absent from the "supported suffixes" list and the failure gives no
    # hint that a dependency is missing.
    # The test env installs every extra, so simulate the missing one on both
    # sides: its suffix drops out of the supported set and shows up as
    # unavailable, exactly as it would with the reader unimportable.
    image_conversion = importlib.import_module(
        "webknossos.dataset._utils.image_conversion"
    )
    available = image_conversion.get_valid_suffixes()
    monkeypatch.setattr(
        image_conversion,
        "get_valid_suffixes",
        lambda: available - {"ims"},
    )
    monkeypatch.setattr(
        image_conversion,
        "get_unavailable_suffixes",
        lambda: {"ims": "ims"},
    )
    (tmp_upath / "a.ims").write_bytes(b"stand-in for an ims file")

    with pytest.raises(
        UnsupportedImageFormatError, match=r"pip install webknossos\[ims\]"
    ) as excinfo:
        Dataset.from_images(tmp_upath, tmp_upath / "ds", voxel_size=(1, 1, 1))
    assert ".ims" in str(excinfo.value)
    # A non-empty missing_extras is how downstream tells "install this" apart
    # from "this format cannot be converted at all".
    assert excinfo.value.missing_extras == ("ims",)


def test_from_images_error_unchanged_when_nothing_is_missing(
    tmp_upath: UPath,
) -> None:
    # The hint must not appear when every reader imported fine.
    (tmp_upath / "a.unsupported").write_bytes(b"x")
    with pytest.raises(
        UnsupportedImageFormatError, match="Could not find any supported image data"
    ) as excinfo:
        Dataset.from_images(tmp_upath, tmp_upath / "ds", voxel_size=(1, 1, 1))
    error = excinfo.value
    assert error.missing_extras == ()
    # The input is a directory, so there is no single offending suffix.
    assert error.suffix is None
    assert error.path == tmp_upath
    assert "tif" in error.supported_suffixes


def test_from_images_single_unsupported_file(tmp_upath: UPath) -> None:
    # Passing a single file that no reader handles used to raise an
    # UnboundLocalError, because input_files was only assigned for files with a
    # supported suffix.
    unsupported = tmp_upath / "scan.dcm"
    unsupported.write_bytes(b"\x00" * 132)

    with pytest.raises(
        UnsupportedImageFormatError, match="Could not find any supported image data"
    ) as excinfo:
        Dataset.from_images(unsupported, tmp_upath / "ds", voxel_size=(1, 1, 1))
    error = excinfo.value
    assert error.suffix == "dcm"
    assert error.path == unsupported
    assert error.missing_extras == ()
    # Subclassing ValueError keeps `except ValueError` callers working.
    assert isinstance(error, ValueError)

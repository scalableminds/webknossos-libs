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
from tests.data_fixtures import (
    create_synthetic_multi_timepoint_ims,
    download_wklibs_sample_archive,
    write_zarr_v3_array,
)
from webknossos.dataset import (
    Dataset,
    RemoteDataset,
    UnsupportedImageFormatError,
)
from webknossos.dataset._image_conversion.image_source import ReadOptions
from webknossos.dataset._image_conversion.mrc_image_source import MrcImageSource
from webknossos.dataset._image_conversion.tiff_slice_reader import TiffSliceReader
from webknossos.geometry import (
    C_AXIS,
    T_AXIS,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
    BoundingBox,
    Vec3Int,
    VecInt,
)


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

    reader = TiffSliceReader(tif_path)
    reader.bundle_axes = [Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]

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
    reader = TiffSliceReader(tif_path)
    reader.bundle_axes = [C_AXIS, Y_AXIS, X_AXIS]
    reader.iter_axes = [Z_AXIS]

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
            download_wklibs_sample_archive("various_tiff_formats"),
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
        "tiffs_test_S.tif": ("uint16", 1, VecInt(c=1, s=3, z=64, y=128, x=128)),
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
    # from_images() passes allow_multiple_layers=True, but the RGB channels of
    # an everyday image format still belong in one layer rather than being
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
    # ImsImageSource.get_layer_split_options() reports {"channel": [0, 1]} and
    # from_images() (which always passes allow_multiple_layers=True) should
    # split it into one layer per channel instead of picking just the first.
    ims_path = download_wklibs_sample_archive("brain_crop3.ims")

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
    # per channel (via get_layer_split_options()'s {"channel": [...]} report),
    # each keeping its own "t" axis for all timepoints, rather than raising
    # the "multiple timepoints and multiple channels" ValueError that firing
    # on the unpinned from_images() discovery probe would otherwise cause.
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
        assert layer.bounding_box.axes == (T_AXIS, C_AXIS, X_AXIS, Y_AXIS, Z_AXIS)
        assert layer.bounding_box.size.to_tuple() == (2, 1, 10, 8, 4)
        data = layer.get_finest_mag().read()  # (t, c, x, y, z)
        for t in range(2):
            assert (data[t, 0] == t * 100 + c).all()


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
    assert reader.get_layer_split_options() is None
    assert reader.expected_bbox.size_xyz.to_tuple() == (X, Y, Z)
    assert reader.expected_bbox.size.c == 1


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
            bbox = BoundingBox((0, 0, z), (X, Y, 1)).normalize_axes(1)
            reader.copy_chunk_to_view(bbox, mag_view=mag_view, dtype=None)

    assert open_count == Z, (
        f"Expected mrcfile.mmap to be called {Z} times (once per chunk), got {open_count}"
    )


def test_no_slashes_in_layername(tmp_upath: UPath) -> None:
    (input_path := tmp_upath / "tiff" / "subfolder" / "tifffiles").mkdir(parents=True)
    copytree(
        str(download_wklibs_sample_archive("tiff_with_different_shapes")),
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


def test_from_images_zarr_directories_are_not_descended_into(tmp_upath: UPath) -> None:
    # A directory tree with a suffixed and a bare Zarr layer, plus a plain
    # TIFF: the two Zarr directories must each become a single input entry —
    # not be walked into for their internal chunk files, and not silently
    # dropped either (the old plain-glob walk did the latter, since none of a
    # Zarr array's own chunk files carry a recognized extension).
    input_dir = tmp_upath / "input"
    input_dir.mkdir(parents=True)

    suffixed = np.arange(4 * 8 * 8, dtype="uint8").reshape(4, 8, 8)
    write_zarr_v3_array(
        input_dir / "layer_a.zarr", suffixed, dimension_names=[Z_AXIS, Y_AXIS, X_AXIS]
    )
    bare = np.arange(4 * 8 * 8, dtype="uint8").reshape(4, 8, 8) + 1
    write_zarr_v3_array(
        input_dir / "layer_b", bare, dimension_names=[Z_AXIS, Y_AXIS, X_AXIS]
    )
    tiff_data = np.arange(8 * 8, dtype="uint8").reshape(8, 8).astype("uint8")
    imwrite(str(input_dir / "layer_c.tif"), tiff_data)

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            input_dir,
            tmp_upath / "ds",
            voxel_size=(1, 1, 1),
            map_filepath_to_layer_name=Dataset.ConversionLayerMapping.ENFORCE_LAYER_PER_FILE,
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"layer_a.zarr", "layer_b", "layer_c.tif"}
    np.testing.assert_array_equal(
        ds.layers["layer_a.zarr"].get_finest_mag().read()[0],
        suffixed.transpose(2, 1, 0),
    )
    np.testing.assert_array_equal(
        ds.layers["layer_b"].get_finest_mag().read()[0], bare.transpose(2, 1, 0)
    )
    np.testing.assert_array_equal(
        ds.layers["layer_c.tif"].get_finest_mag().read()[0, :, :, 0], tiff_data.T
    )


def test_from_images_input_path_is_itself_a_store_directory(tmp_upath: UPath) -> None:
    # Pointing from_images() directly at a chunked store's root — rather than
    # at a parent directory containing it — must still work: input_upath
    # cannot be walked with _iter_convertible_paths, which only recognizes a
    # store directory among another directory's children, never the root it
    # was called with.
    data = np.arange(4 * 8 * 8, dtype="uint8").reshape(4, 8, 8)
    zarr_path = tmp_upath / "test.zarr"
    write_zarr_v3_array(zarr_path, data, dimension_names=[Z_AXIS, Y_AXIS, X_AXIS])

    with SequentialExecutor() as executor:
        ds = Dataset.from_images(
            zarr_path,
            tmp_upath / "ds",
            voxel_size=(1, 1, 1),
            executor=executor,
        )

    assert set(ds.layers.keys()) == {"test.zarr"}
    np.testing.assert_array_equal(
        ds.layers["test.zarr"].get_finest_mag().read()[0], data.transpose(2, 1, 0)
    )


def test_valid_chunked_image_source_extensions_include_zarr_and_n5() -> None:
    from webknossos.dataset._image_conversion.image_source_registry import (
        get_valid_chunked_image_source_extensions,
    )

    extensions = get_valid_chunked_image_source_extensions()
    assert "zarr" in extensions
    assert "n5" in extensions


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


def test_optional_reader_extensions_match_supported_file_extensions() -> None:
    # _OPTIONAL_SLICE_READERS_AND_IMAGE_SOURCES has to restate each reader's
    # extensions, because a
    # reader whose dependency is missing never imports and so cannot report
    # its own supported_file_extensions(). Whenever a reader *is* importable,
    # the two must agree — otherwise the missing-dependency hint names the
    # wrong formats, or silently stops covering one. Covers both strategies:
    # slice readers are just as optional as chunked ones now that tifffile is
    # not in the base install.
    from webknossos.dataset._image_conversion.chunked_image_source import (
        ChunkedImageSource,
    )
    from webknossos.dataset._image_conversion.image_source_registry import (
        _CHUNKED_IMAGE_SOURCE_CLASSES,
        _OPTIONAL_SLICE_READERS_AND_IMAGE_SOURCES,
        _SLICE_READER_CLASSES,
        get_unavailable_extensions,
    )
    from webknossos.dataset._image_conversion.slice_reader import SliceReader

    # Annotated because the two lists' only common base is ABC, which does
    # not declare supported_file_extensions(); the union does.
    registered: dict[str, type[SliceReader] | type[ChunkedImageSource]] = {
        cls.__name__: cls for cls in _SLICE_READER_CLASSES
    }
    registered.update({cls.__name__: cls for cls in _CHUNKED_IMAGE_SOURCE_CLASSES})
    # The test env installs every extra except czi on Python 3.14, since
    # pylibCZIrw ships no wheel there.
    unavailable_extras = set(get_unavailable_extensions().values())
    for reader in _OPTIONAL_SLICE_READERS_AND_IMAGE_SOURCES:
        if reader.extra in unavailable_extras:
            continue
        assert reader.class_name in registered, (
            f"{reader.class_name} is declared optional but did not register"
        )
        assert registered[reader.class_name].supported_file_extensions() == set(
            reader.extensions
        ), (
            f"declared extensions for {reader.class_name} are out of sync "
            "with its supported_file_extensions()"
        )


def test_from_images_names_missing_optional_dependency(
    tmp_upath: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With an extra uninstalled its reader never registers, so its formats are
    # simply absent from the "supported extensions" list and the failure gives
    # no hint that a dependency is missing.
    # The test env installs every extra, so simulate the missing one on both
    # sides: its extension drops out of the supported set and shows up as
    # unavailable, exactly as it would with the reader unimportable.
    image_conversion = importlib.import_module(
        "webknossos.dataset._image_conversion.image_conversion"
    )
    available = image_conversion.get_valid_extensions()
    monkeypatch.setattr(
        image_conversion,
        "get_valid_extensions",
        lambda: available - {"ims"},
    )
    monkeypatch.setattr(
        image_conversion,
        "get_unavailable_extensions",
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
    # The input is a directory, so there is no single offending extension.
    assert error.file_extension is None
    assert error.path == tmp_upath
    assert "tif" in error.supported_file_extensions


def test_from_images_single_unsupported_file(tmp_upath: UPath) -> None:
    # Passing a single file that no reader handles used to raise an
    # UnboundLocalError, because input_files was only assigned for files with a
    # supported extension.
    unsupported = tmp_upath / "scan.dcm"
    unsupported.write_bytes(b"\x00" * 132)

    with pytest.raises(
        UnsupportedImageFormatError, match="Could not find any supported image data"
    ) as excinfo:
        Dataset.from_images(unsupported, tmp_upath / "ds", voxel_size=(1, 1, 1))
    error = excinfo.value
    assert error.file_extension == "dcm"
    assert error.path == unsupported
    assert error.missing_extras == ()
    # Subclassing ValueError keeps `except ValueError` callers working.
    assert isinstance(error, ValueError)

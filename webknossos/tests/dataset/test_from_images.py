import json
import warnings
from collections.abc import Iterator
from shutil import copytree
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cluster_tools import SequentialExecutor
from tifffile import TiffFile, imwrite
from upath import UPath

from tests.constants import TESTDATA_DIR
from webknossos.dataset import Dataset, RemoteDataset
from webknossos.dataset._utils.pims_tiff_reader import PimsTiffReader
from webknossos.geometry import Vec3Int, VecInt


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
    reader = PimsTiffReader(tif_path)
    reader.bundle_axes = ["c", "y", "x"]
    reader.iter_axes = ["z"]

    pages_read: list[int] = []
    original_asarray = tifffile_module.TiffPage.asarray

    def tracking_asarray(self: tifffile_module.TiffPage, **kwargs: Any) -> np.ndarray:
        pages_read.append(self.index)
        return original_asarray(self, **kwargs)

    with patch.object(tifffile_module.TiffPage, "asarray", tracking_asarray):
        frame_z0 = np.array(reader[0])

    assert pages_read == [0, 2, 4], (
        f"Expected pages [0, 2, 4] for z=0, got {pages_read}"
    )
    assert frame_z0.shape == (C, Y, X)
    np.testing.assert_array_equal(frame_z0, data[:, 0, :, :])


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


def test_from_dicom_images(tmp_upath: UPath) -> None:
    ds = Dataset.from_images(
        TESTDATA_DIR / "dicoms",
        tmp_upath,
        (1, 1, 1),
        use_bioformats=True,
    )
    assert len(ds.layers) == 1
    assert "dicoms" in ds.layers
    data = ds.layers["dicoms"].get_finest_mag().read()
    assert data.shape == (1, 274, 384, 10)
    assert data.max() == 110, (
        f"The maximum value of the image should be 127 but is {data.max()}"
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

import json
import warnings
from collections.abc import Iterator
from shutil import copytree

import numpy as np
import pytest
from cluster_tools import SequentialExecutor
from tifffile import TiffFile
from upath import UPath

from tests.constants import TESTDATA_DIR
from webknossos.dataset import Dataset


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
        "tiffs_test_CS.tif__channel0": ("uint8", 1, (3, 64, 128, 128)),
        "tiffs_test_CS.tif__channel1": ("uint8", 1, (3, 64, 128, 128)),
        "tiffs_test_CS.tif__channel2": ("uint8", 1, (3, 64, 128, 128)),
        "tiffs_test_CS.tif__channel3": ("uint8", 1, (3, 64, 128, 128)),
        "tiffs_test_CS.tif__channel4": ("uint8", 1, (3, 64, 128, 128)),
        "tiffs_test_C.tif__channel0": ("uint8", 1, (128, 128, 64)),
        "tiffs_test_C.tif__channel1": ("uint8", 1, (128, 128, 64)),
        "tiffs_test_C.tif__channel2": ("uint8", 1, (128, 128, 64)),
        "tiffs_test_C.tif__channel3": ("uint8", 1, (128, 128, 64)),
        "tiffs_test_C.tif__channel4": ("uint8", 1, (128, 128, 64)),
        "tiffs_test_I.tif": ("uint32", 1, (64, 128, 64)),
        "tiffs_test_S.tif": ("uint16", 1, (3, 64, 128, 128)),
    }

    for layer_name, layer in ds.layers.items():
        dtype, channels, size = expected_dtype_channels_size_per_layer[layer_name]
        assert layer.dtype_per_channel == np.dtype(dtype)
        assert layer.num_channels == channels
        assert layer.bounding_box.size == size

        # Check that the zarr.json metadata is correct
        mag1 = layer.get_finest_mag()
        array_shape = json.loads((mag1.path / "zarr.json").read_bytes())["shape"]
        shard_aligned_bottomright = layer.bounding_box.with_bottomright_xyz(
            layer.bounding_box.bottomright_xyz.ceildiv(mag1.info.shard_shape)
            * mag1.info.shard_shape
        ).bottomright
        assert array_shape == [channels] + shard_aligned_bottomright.to_list()


def test_from_dicom_images(tmp_upath: UPath) -> None:
    ds = Dataset.from_images(
        TESTDATA_DIR / "dicoms",
        tmp_upath,
        (1, 1, 1),
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

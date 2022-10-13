from pathlib import Path

import numpy as np
from tifffile import TiffFile

import webknossos as wk


def test_compare_tifffile(tmp_path: Path) -> None:
    ds = wk.Dataset.from_images(
        "testdata/tiff",
        tmp_path,
        (1, 1, 1),
        compress=True,
        layer_category="segmentation",
        map_filepath_to_layer_name=wk.Dataset.ConversionLayerMapping.ENFORCE_SINGLE_LAYER,
    )
    assert len(ds.layers) == 1
    assert "tiff" in ds.layers
    data = ds.layers["tiff"].get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with TiffFile("testdata/tiff/test.0000.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        assert np.array_equal(data[:, :, z_index], comparison_slice)


def test_multiple_multitiffs(tmp_path: Path) -> None:
    ds = wk.Dataset.from_images(
        "testdata/various_tiff_formats",
        tmp_path,
        (1, 1, 1),
        compress=True,
    )
    assert len(ds.layers) == 4

    expected_dtype_channels_size_per_layer = {
        "test_CS.tif": ("uint8", 3, (128, 128, 320)),
        "test_C.tif": ("uint8", 1, (128, 128, 320)),
        "test_I.tif": ("uint32", 1, (64, 128, 64)),
        "test_S.tif": ("uint16", 3, (128, 128, 64)),
    }

    for layer_name, layer in ds.layers.items():
        dtype, channels, size = expected_dtype_channels_size_per_layer[layer_name]
        assert layer.dtype_per_channel == np.dtype(dtype)
        assert layer.num_channels == channels
        assert layer.bounding_box.size == size

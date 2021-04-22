import filecmp
import itertools
import json
import os
import tempfile
from os.path import dirname, join
from pathlib import Path
from typing import Any, Tuple, cast, Generator

import pytest

import numpy as np
from shutil import rmtree, copytree

from scipy.ndimage import zoom
from wkw import wkw
from wkw.wkw import WKWException

from wkcuber.api import Dataset
from wkcuber.api.Dataset import (
    WKDataset,
    TiffDataset,
    TiledTiffDataset,
    AbstractDataset,
)
from os import path, makedirs

from wkcuber.api.Layer import Layer
from wkcuber.api.MagDataset import (
    find_mag_path_on_disk,
    MagDataset,
    TiledTiffMagDataset,
)
from wkcuber.api.Properties.DatasetProperties import TiffProperties, WKProperties
from wkcuber.api.TiffData.TiffMag import TiffReader
from wkcuber.api.View import View
from wkcuber.api.bounding_box import BoundingBox
from wkcuber.compress import compress_mag_inplace
from wkcuber.downsampling import downsample_mags_isotropic, downsample_mags_anisotropic
from wkcuber.downsampling_utils import (
    downsample_cube,
    InterpolationModes,
    parse_interpolation_mode,
)
from wkcuber.mag import Mag
from wkcuber.utils import get_executor_for_args, open_wkw, WkwDatasetInfo, named_partial

expected_error_msg = "The test did not throw an exception even though it should. "


def delete_dir(relative_path: str) -> None:
    if path.exists(relative_path) and path.isdir(relative_path):
        rmtree(relative_path)


def chunk_job(args: Tuple[View, int]) -> None:
    (view, i) = args
    # increment the color value of each voxel
    data = view.read(size=view.size)
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    view.write(data)


def advanced_chunk_job(args: Tuple[View, int], type: type) -> None:
    view, i = args

    # write different data for each chunk (depending on the global_offset of the chunk)
    data = view.read(size=view.size)
    data = np.ones(data.shape, dtype=type) * type(sum(view.global_offset))
    view.write(data)


def for_each_chunking_with_wrong_chunk_size(view: View) -> None:
    with get_executor_for_args(None) as executor:
        try:
            view.for_each_chunk(
                chunk_job,
                chunk_size=(0, 64, 64),
                executor=executor,
            )
            raise Exception(
                expected_error_msg + "The chunk_size should not contain zeros"
            )
        except AssertionError:
            pass

        try:
            view.for_each_chunk(
                chunk_job,
                chunk_size=(16, 64, 64),
                executor=executor,
            )
            raise Exception(expected_error_msg)
        except AssertionError:
            pass

        try:
            view.for_each_chunk(
                chunk_job,
                chunk_size=(100, 64, 64),
                executor=executor,
            )
            raise Exception(expected_error_msg)
        except AssertionError:
            pass


def for_each_chunking_advanced(ds: AbstractDataset, view: View) -> None:
    chunk_size = (64, 64, 64)
    with get_executor_for_args(None) as executor:
        func = named_partial(advanced_chunk_job, type=np.uint8)
        view.for_each_chunk(
            func,
            chunk_size=chunk_size,
            executor=executor,
        )

    for offset, size in [
        ((10, 10, 10), (54, 54, 54)),
        ((10, 64, 10), (54, 64, 54)),
        ((10, 128, 10), (54, 32, 54)),
        ((64, 10, 10), (64, 54, 54)),
        ((64, 64, 10), (64, 64, 54)),
        ((64, 128, 10), (64, 32, 54)),
        ((128, 10, 10), (32, 54, 54)),
        ((128, 64, 10), (32, 64, 54)),
        ((128, 128, 10), (32, 32, 54)),
    ]:
        chunk = ds.get_view("color", "1", size=size, offset=offset, is_bounded=False)
        chunk_data = chunk.read(size=chunk.size)
        assert np.array_equal(
            np.ones(chunk_data.shape, dtype=np.uint8)
            * np.uint8(sum(chunk.global_offset)),
            chunk_data,
        )


def copy_and_transform_job(args: Tuple[View, View, int], name: str, val: int) -> None:
    (source_view, target_view, i) = args
    # This method simply takes the data from the source_view, transforms it and writes it to the target_view

    # These assertions are just to demonstrate how the passed parameters can be accessed inside this method
    assert name == "foo"
    assert val == 42

    # increment the color value of each voxel
    data = source_view.read(size=source_view.size)
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    target_view.write(data)


def get_multichanneled_data(dtype: type) -> np.ndarray:
    data = np.zeros((3, 250, 200, 10), dtype=dtype)
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256
    return data


def test_create_wk_dataset_with_layer_and_mag() -> None:
    delete_dir("./testoutput/wk_dataset")

    ds = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("./testoutput/wk_dataset/color/1")
    assert path.exists("./testoutput/wk_dataset/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2


def test_create_wk_dataset_with_explicit_header_fields() -> None:
    delete_dir("./testoutput/wk_dataset_advanced")

    ds = WKDataset.create("./testoutput/wk_dataset_advanced", scale=(1, 1, 1))
    ds.add_layer("color", Layer.COLOR_TYPE, dtype_per_layer="uint48", num_channels=3)

    ds.get_layer("color").add_mag("1", block_len=64, file_len=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("./testoutput/wk_dataset_advanced/color/1")
    assert path.exists("./testoutput/wk_dataset_advanced/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2

    assert ds.properties.data_layers["color"].element_class == "uint48"
    assert (
        ds.properties.data_layers["color"].wkw_magnifications[0].cube_length == 64 * 64
    )  # mag "1"
    assert ds.properties.data_layers["color"].wkw_magnifications[0].mag == Mag("1")
    assert (
        ds.properties.data_layers["color"].wkw_magnifications[1].cube_length == 32 * 32
    )  # mag "2-2-1" (defaults are used)
    assert ds.properties.data_layers["color"].wkw_magnifications[1].mag == Mag("2-2-1")


def test_create_tiff_dataset_with_layer_and_mag() -> None:
    # This test would be the same for WKDataset

    delete_dir("./testoutput/tiff_dataset")

    ds = WKDataset.create("./testoutput/tiff_dataset", scale=(1, 1, 1))
    ds.add_layer("color", Layer.COLOR_TYPE)

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("./testoutput/tiff_dataset/color/1")
    assert path.exists("./testoutput/tiff_dataset/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2


def test_open_wk_dataset() -> None:
    ds = WKDataset("./testdata/simple_wk_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1


def test_open_tiff_dataset() -> None:
    ds = TiffDataset("./testdata/simple_tiff_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1


def test_view_read_with_open() -> None:

    wk_view = WKDataset("./testdata/simple_wk_dataset/").get_view(
        "color", "1", size=(16, 16, 16)
    )

    assert not wk_view._is_opened

    with wk_view.open():
        assert wk_view._is_opened

        data = wk_view.read(size=(10, 10, 10))
        assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_tiff_mag_read_with_open() -> None:

    tiff_dataset = TiffDataset("./testdata/simple_tiff_dataset/")
    layer = tiff_dataset.get_layer("color")
    mag = layer.get_mag("1")
    mag.open()
    data = mag.read(size=(10, 10, 10))
    assert data.shape == (1, 10, 10, 10)  # single channel


def test_view_read_without_open() -> None:
    # This test would be the same for TiffDataset

    wk_view = WKDataset("./testdata/simple_wk_dataset/").get_view(
        "color", "1", size=(16, 16, 16)
    )

    assert not wk_view._is_opened

    # 'read()' checks if it was already opened. If not, it opens and closes automatically
    data = wk_view.read(size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_view_wk_write() -> None:
    delete_dir("./testoutput/simple_wk_dataset/")
    copytree("./testdata/simple_wk_dataset/", "./testoutput/simple_wk_dataset/")

    wk_view = WKDataset("./testoutput/simple_wk_dataset/").get_view(
        "color", "1", size=(16, 16, 16)
    )

    with wk_view.open():
        np.random.seed(1234)
        write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

        wk_view.write(write_data)

        data = wk_view.read(size=(10, 10, 10))
        assert np.array_equal(data, write_data)


def test_view_tiff_write() -> None:
    delete_dir("./testoutput/simple_tiff_dataset/")
    copytree("./testdata/simple_tiff_dataset/", "./testoutput/simple_tiff_dataset/")

    tiff_view = TiffDataset("./testoutput/simple_tiff_dataset/").get_view(
        "color", "1", size=(16, 16, 10)
    )

    with tiff_view.open():
        np.random.seed(1234)
        write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)

        tiff_view.write(write_data)

        data = tiff_view.read(size=(5, 5, 5))
        assert data.shape == (1, 5, 5, 5)  # this dataset has only one channel
        assert np.array_equal(data, np.expand_dims(write_data, 0))


def test_view_tiff_write_out_of_bounds() -> None:
    new_dataset_path = "./testoutput/tiff_view_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_tiff_dataset/", new_dataset_path)

    tiff_view = TiffDataset(new_dataset_path).get_view(
        "color", "1", size=(100, 100, 10)
    )

    with tiff_view.open():
        try:
            tiff_view.write(
                np.zeros((200, 200, 5), dtype=np.uint8)
            )  # this is bigger than the bounding_box
            raise Exception(
                "The test 'test_view_tiff_write_out_of_bounds' did not throw an exception even though it should"
            )
        except AssertionError:
            pass


def test_view_wk_write_out_of_bounds() -> None:
    new_dataset_path = "./testoutput/wk_view_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_wk_dataset/", new_dataset_path)

    tiff_view = WKDataset(new_dataset_path).get_view("color", "1", size=(16, 16, 16))

    with tiff_view.open():
        try:
            tiff_view.write(
                np.zeros((200, 200, 5), dtype=np.uint8)
            )  # this is bigger than the bounding_box
            raise Exception(
                "The test 'test_view_wk_write_out_of_bounds' did not throw an exception even though it should"
            )
        except AssertionError:
            pass


def test_wk_view_out_of_bounds() -> None:
    try:
        # The size of the mag is (24, 24, 24). Trying to get an bigger view should throw an error
        WKDataset("./testdata/simple_wk_dataset/").get_view(
            "color", "1", size=(100, 100, 100)
        )
        raise Exception(
            "The test 'test_view_wk_write_out_of_bounds' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_tiff_view_out_of_bounds() -> None:
    try:
        # The size of the mag is (24, 24, 24). Trying to get an bigger view should throw an error
        TiffDataset("./testdata/simple_tiff_dataset/").get_view(
            "color", "1", size=(100, 100, 100)
        )
        raise Exception(
            "The test 'test_view_wk_write_out_of_bounds' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_tiff_write_out_of_bounds() -> None:
    new_dataset_path = "./testoutput/simple_tiff_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_tiff_dataset/", new_dataset_path)

    ds = TiffDataset(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (265, 265, 10)
    mag_dataset.write(
        np.zeros((300, 300, 15), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (300, 300, 15)


def test_wk_write_out_of_bounds() -> None:
    new_dataset_path = "./testoutput/simple_wk_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_wk_dataset/", new_dataset_path)

    ds = WKDataset(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 24)
    mag_dataset.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 48)


def test_wk_write_out_of_bounds_mag2() -> None:
    new_dataset_path = "./testoutput/simple_wk_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_wk_dataset/", new_dataset_path)

    ds = WKDataset(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_or_add_mag("2-2-1")

    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 24)
    mag_dataset.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8), (10, 10, 10)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (120, 24, 58)


def test_update_new_bounding_box_offset() -> None:
    # This test would be the same for WKDataset

    delete_dir("./testoutput/tiff_dataset")

    ds = TiffDataset.create("./testoutput/tiff_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")

    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (-1, -1, -1)

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(
        write_data, offset=(10, 10, 10)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (10, 10, 10)

    mag.write(
        write_data, offset=(5, 5, 20)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (5, 5, 10)


def test_other_file_extensions_for_tiff_dataset() -> None:
    # The TiffDataset also works with other file extensions (in this case .png)
    # It also works with .jpg but this format uses lossy compression

    delete_dir("./testoutput/png_dataset")

    ds = TiffDataset.create(
        "./testoutput/png_dataset", scale=(1, 1, 1), pattern="{zzz}.png"
    )
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data)
    assert np.array_equal(mag.read(size=(10, 10, 10)), np.expand_dims(write_data, 0))


def test_tiff_write_multi_channel_uint8() -> None:
    dataset_path = "./testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag("1")

    # 10 images (z-layers), each 250x200, dtype=np.uint8
    data = get_multichanneled_data(np.uint8)

    ds_tiff.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 200, 10)))


def test_wk_write_multi_channel_uint8() -> None:
    dataset_path = "./testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag("1")

    # 10 images (z-layers), each 250x200, dtype=np.uint8
    data = get_multichanneled_data(np.uint8)

    ds_tiff.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 200, 10)))


def test_tiff_write_multi_channel_uint16() -> None:
    dataset_path = "./testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer(
        "color", Layer.COLOR_TYPE, num_channels=3, dtype_per_layer="uint48"
    ).add_mag("1")

    # 10 images (z-layers), each 250x200, dtype=np.uint16
    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read(size=(250, 200, 10))

    print(written_data.dtype)

    assert np.array_equal(data, written_data)


def test_wk_write_multi_channel_uint16() -> None:
    dataset_path = "./testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer(
        "color", Layer.COLOR_TYPE, num_channels=3, dtype_per_layer="uint48"
    ).add_mag("1")

    # 10 images (z-layers), each 250x200, dtype=np.uint16
    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read(size=(250, 200, 10))

    assert np.array_equal(data, written_data)


def test_wkw_empty_read() -> None:
    filename = "./testoutput/empty_wk_dataset"
    delete_dir(filename)

    mag = (
        WKDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE)
        .add_mag("1")
    )
    data = mag.read(offset=(1, 1, 1), size=(0, 0, 0))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_empty_read() -> None:
    filename = "./testoutput/empty_tiff_dataset"
    delete_dir(filename)

    mag = (
        TiffDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE)
        .add_mag("1")
    )
    data = mag.read(offset=(1, 1, 1), size=(0, 0, 0))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_read_padded_data() -> None:
    filename = "./testoutput/empty_tiff_dataset"
    delete_dir(filename)

    mag = (
        TiffDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE, num_channels=3)
        .add_mag("1")
    )
    # there are no tiffs yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_wk_read_padded_data() -> None:
    filename = "./testoutput/empty_wk_dataset"
    delete_dir(filename)

    mag = (
        WKDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE, num_channels=3)
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_read_and_write_of_properties() -> None:
    destination_path = "./testoutput/read_write_properties/"
    delete_dir(destination_path)
    source_file_name = "./testdata/simple_tiff_dataset/datasource-properties.json"
    destination_file_name = destination_path + "datasource-properties.json"

    imported_properties = TiffProperties._from_json(source_file_name)
    imported_properties._path = destination_file_name
    makedirs(destination_path)
    imported_properties._export_as_json()

    filecmp.cmp(source_file_name, destination_file_name)


def test_num_channel_mismatch_assertion() -> None:
    delete_dir("./testoutput/wk_dataset")

    ds = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", Layer.COLOR_TYPE, num_channels=1).add_mag(
        "1"
    )  # num_channel=1 is also the default

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)  # 3 channels

    try:
        mag.write(write_data)  # there is a mismatch between the number of channels
        raise Exception(
            "The test 'test_num_channel_mismatch_assertion' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_layer() -> None:
    # This test would be the same for TiffDataset

    delete_dir("./testoutput/wk_dataset")

    ds = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color", Layer.COLOR_TYPE, dtype_per_layer="uint8", num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color", Layer.COLOR_TYPE, dtype_per_layer="uint8", num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    try:
        # layer did exist before but with another 'dtype_per_layer' (this would work the same for 'category' and 'num_channels')
        layer = ds.get_or_add_layer(
            "color", Layer.COLOR_TYPE, dtype_per_layer="uint16", num_channels=1
        )

        raise Exception(
            "The test 'test_get_or_add_layer' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_mag_for_wk() -> None:
    delete_dir("./testoutput/wk_dataset")

    layer = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1)).add_layer(
        "color", Layer.COLOR_TYPE
    )

    assert "1" not in layer.mags.keys()

    # The mag did not exist before
    mag = layer.get_or_add_mag("1", block_len=32, file_len=32, block_type=1)
    assert "1" in layer.mags.keys()
    assert mag.name == "1"

    # The mag did exist before
    layer.get_or_add_mag("1", block_len=32, file_len=32, block_type=1)
    assert "1" in layer.mags.keys()
    assert mag.name == "1"

    try:
        # mag did exist before but with another 'block_len' (this would work the same for 'file_len' and 'block_type')
        mag = layer.get_or_add_mag("1", block_len=64, file_len=32, block_type=1)

        raise Exception(
            "The test 'test_get_or_add_layer' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_mag_for_tiff() -> None:
    delete_dir("./testoutput/wk_dataset")

    layer = TiffDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1)).add_layer(
        "color", Layer.COLOR_TYPE
    )

    assert "1" not in layer.mags.keys()

    # The mag did not exist before
    mag = layer.get_or_add_mag("1")
    assert "1" in layer.mags.keys()
    assert mag.name == "1"

    # The mag did exist before
    layer.get_or_add_mag("1")
    assert "1" in layer.mags.keys()
    assert mag.name == "1"


def test_tiled_tiff_read_and_write_multichannel() -> None:
    delete_dir("./testoutput/TiledTiffDataset")
    tiled_tiff_ds = TiledTiffDataset.create(
        "./testoutput/TiledTiffDataset",
        scale=(1, 1, 1),
        tile_size=(32, 64),
        pattern="{xxx}_{yyy}_{zzz}.tif",
    )

    mag = tiled_tiff_ds.add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag(
        "1"
    )

    data = get_multichanneled_data(np.uint8)

    mag.write(data, offset=(5, 5, 5))
    written_data = mag.read(offset=(5, 5, 5), size=(250, 200, 10))
    assert written_data.shape == (3, 250, 200, 10)
    assert np.array_equal(data, written_data)


def test_tiled_tiff_read_and_write() -> None:
    delete_dir("./testoutput/tiled_tiff_dataset")
    tiled_tiff_ds = TiledTiffDataset.create(
        "./testoutput/tiled_tiff_dataset",
        scale=(1, 1, 1),
        tile_size=(32, 64),
        pattern="{xxx}_{yyy}_{zzz}.tif",
    )

    mag = tiled_tiff_ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")

    data = np.zeros((250, 200, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[i, j, h] = i + j % 250

    mag.write(data, offset=(5, 5, 5))
    written_data = mag.read(offset=(5, 5, 5), size=(250, 200, 10))
    assert written_data.shape == (1, 250, 200, 10)
    assert np.array_equal(written_data, np.expand_dims(data, 0))

    assert mag.get_tile(1, 1, 6).shape == (1, 32, 64, 1)
    assert np.array_equal(
        mag.get_tile(1, 2, 6)[0, :, :, 0],
        TiffReader("./testoutput/tiled_tiff_dataset/color/1/001_002_006.tif").read(),
    )

    assert np.array_equal(
        data[(32 * 1) - 5 : (32 * 2) - 5, (64 * 2) - 5 : (64 * 3) - 5, 6],
        TiffReader("./testoutput/tiled_tiff_dataset/color/1/001_002_006.tif").read(),
    )


def test_open_dataset_without_num_channels_in_properties() -> None:
    delete_dir("./testoutput/old_wk_dataset/")
    copytree("./testdata/old_wk_dataset/", "./testoutput/old_wk_dataset/")

    with open(
        "./testoutput/old_wk_dataset/datasource-properties.json"
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") is None

    ds = WKDataset("./testoutput/old_wk_dataset/")
    assert ds.properties.data_layers["color"].num_channels == 1
    ds.properties._export_as_json()

    with open(
        "./testoutput/old_wk_dataset/datasource-properties.json"
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") == 1


def test_advanced_pattern() -> None:
    delete_dir("./testoutput/tiff_dataset_advanced_pattern")
    ds = TiledTiffDataset.create(
        "./testoutput/tiff_dataset_advanced_pattern",
        scale=(1, 1, 1),
        tile_size=(32, 32),
        pattern="{xxxx}/{yyyy}/{zzzz}.tif",
    )
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")
    data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(data)

    assert np.array_equal(mag.read(size=(10, 10, 10)), np.expand_dims(data, 0))


def test_invalid_pattern() -> None:

    delete_dir("./testoutput/tiff_invalid_dataset")
    try:
        TiledTiffDataset.create(
            "./testoutput/tiff_invalid_dataset",
            scale=(1, 1, 1),
            tile_size=(32, 32),
            pattern="{xxxx}/{yyyy}/{zzzz.tif",
        )
        raise Exception(
            "The test 'test_invalid_pattern' did not throw an exception even though it should"
        )
    except AssertionError:
        pass

    try:
        TiledTiffDataset.create(
            "./testoutput/tiff_invalid_dataset",
            scale=(1, 1, 1),
            tile_size=(32, 32),
            pattern="zzzz.tif",
        )
        raise Exception(
            "The test 'test_invalid_pattern' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_largest_segment_id_requirement() -> None:
    path = "./testoutput/largest_segment_id"
    delete_dir(path)
    ds = WKDataset.create(path, scale=(10, 10, 10))

    with pytest.raises(AssertionError):
        ds.add_layer("segmentation", Layer.SEGMENTATION_TYPE)

    largest_segment_id = 10
    ds.add_layer(
        "segmentation", Layer.SEGMENTATION_TYPE, largest_segment_id=largest_segment_id
    ).add_mag(Mag(1))

    ds = WKDataset(path)
    assert (
        ds.properties.data_layers["segmentation"].largest_segment_id
        == largest_segment_id
    )


def test_properties_with_segmentation() -> None:
    input_json_path = "./testdata/complex_property_ds/datasource-properties.json"
    output_json_path = "./testoutput/complex_property_ds/datasource-properties.json"
    properties = WKProperties._from_json(input_json_path)

    # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
    assert properties.data_layers["segmentation"].largest_segment_id == 1000000000
    assert properties.data_layers["segmentation"].mappings == [
        "larger5um1",
        "axons",
        "astrocyte-ge-7",
        "astrocyte",
        "mitochondria",
        "astrocyte-full",
    ]

    # export the json under a new name
    makedirs(dirname(output_json_path), exist_ok=True)
    properties._path = output_json_path
    properties._export_as_json()

    # validate if contents match
    with open(input_json_path) as input_properties:
        input_data = json.load(input_properties)

        with open(output_json_path) as output_properties:
            output_data = json.load(output_properties)
            for layer in output_data["dataLayers"]:
                # remove the num_channels because they are not part of the original json
                del layer["num_channels"]

            assert input_data == output_data


def test_chunking_wk() -> None:
    delete_dir("./testoutput/chunking_dataset_wk/")
    copytree("./testdata/simple_wk_dataset/", "./testoutput/chunking_dataset_wk/")

    view = WKDataset("./testoutput/chunking_dataset_wk/").get_view(
        "color", "1", size=(256, 256, 256), is_bounded=False
    )

    original_data = view.read(size=view.size)

    with get_executor_for_args(None) as executor:
        view.for_each_chunk(
            chunk_job,
            chunk_size=(64, 64, 64),
            executor=executor,
        )

    assert np.array_equal(original_data + 50, view.read(size=view.size))


def test_chunking_wk_advanced() -> None:
    delete_dir("./testoutput/chunking_dataset_wk_advanced/")
    copytree(
        "./testdata/simple_wk_dataset/", "./testoutput/chunking_dataset_wk_advanced/"
    )

    ds = WKDataset("./testoutput/chunking_dataset_wk_advanced/")
    view = ds.get_view(
        "color", "1", size=(150, 150, 54), offset=(10, 10, 10), is_bounded=False
    )
    for_each_chunking_advanced(ds, view)


def test_chunking_wk_wrong_chunk_size() -> None:
    delete_dir("./testoutput/chunking_dataset_wk_with_wrong_chunk_size/")
    copytree(
        "./testdata/simple_wk_dataset/",
        "./testoutput/chunking_dataset_wk_with_wrong_chunk_size/",
    )

    view = WKDataset(
        "./testoutput/chunking_dataset_wk_with_wrong_chunk_size/"
    ).get_view("color", "1", size=(256, 256, 256), is_bounded=False)

    for_each_chunking_with_wrong_chunk_size(view)


def test_chunking_tiff() -> None:
    delete_dir("./testoutput/chunking_dataset_tiff/")
    copytree("./testdata/simple_tiff_dataset/", "./testoutput/chunking_dataset_tiff/")

    view = TiffDataset("./testoutput/chunking_dataset_tiff/").get_view(
        "color", "1", size=(265, 265, 10)
    )

    original_data = view.read(size=view.size)

    with get_executor_for_args(None) as executor:
        view.for_each_chunk(
            chunk_job,
            chunk_size=(265, 265, 1),
            executor=executor,
        )

    new_data = view.read(size=view.size)
    assert np.array_equal(original_data + 50, new_data)


def test_chunking_tiff_wrong_chunk_size() -> None:
    delete_dir("./testoutput/chunking_dataset_tiff_with_wrong_chunk_size/")
    copytree(
        "./testdata/simple_tiff_dataset/",
        "./testoutput/chunking_dataset_tiff_with_wrong_chunk_size/",
    )

    view = TiffDataset(
        "./testoutput/chunking_dataset_tiff_with_wrong_chunk_size/"
    ).get_view("color", "1", size=(256, 256, 256), is_bounded=False)

    for_each_chunking_with_wrong_chunk_size(view)


def test_chunking_tiled_tiff_wrong_chunk_size() -> None:
    delete_dir("./testoutput/chunking_dataset_tiled_tiff_with_wrong_chunk_size/")

    ds = TiledTiffDataset.create(
        "./testoutput/chunking_dataset_tiled_tiff_with_wrong_chunk_size/",
        scale=(1, 1, 1),
        tile_size=(32, 32),
        pattern="{xxxx}/{yyyy}/{zzzz}.tif",
    )
    ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")
    view = ds.get_view("color", "1", size=(256, 256, 256), is_bounded=False)

    for_each_chunking_with_wrong_chunk_size(view)


def test_chunking_tiled_tiff_advanced() -> None:
    delete_dir("./testoutput/chunking_dataset_tiled_tiff_advanced/")
    copytree(
        "./testdata/simple_wk_dataset/",
        "./testoutput/chunking_dataset_tiled_tiff_advanced/",
    )

    ds = WKDataset("./testoutput/chunking_dataset_tiled_tiff_advanced/")
    view = ds.get_view(
        "color", "1", size=(150, 150, 54), offset=(10, 10, 10), is_bounded=False
    )

    for_each_chunking_advanced(ds, view)


def test_tiled_tiff_inverse_pattern() -> None:
    delete_dir("./testoutput/tiled_tiff_dataset_inverse")
    tiled_tiff_ds = TiledTiffDataset.create(
        "./testoutput/tiled_tiff_dataset_inverse",
        scale=(1, 1, 1),
        tile_size=(32, 64),
        pattern="{zzz}/{xxx}/{yyy}.tif",
    )

    mag = cast(
        TiledTiffMagDataset,
        tiled_tiff_ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1"),
    )

    data = np.zeros((250, 200, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[i, j, h] = i + j % 250

    mag.write(data, offset=(5, 5, 5))
    written_data = mag.read(offset=(5, 5, 5), size=(250, 200, 10))
    assert written_data.shape == (1, 250, 200, 10)
    assert np.array_equal(written_data, np.expand_dims(data, 0))

    assert mag.get_tile(1, 1, 6).shape == (1, 32, 64, 1)
    assert np.array_equal(
        mag.get_tile(1, 2, 6)[0, :, :, 0],
        TiffReader(
            "./testoutput/tiled_tiff_dataset_inverse/color/1/006/001/002.tif"
        ).read(),
    )

    assert np.array_equal(
        data[(32 * 1) - 5 : (32 * 2) - 5, (64 * 2) - 5 : (64 * 3) - 5, 6],
        TiffReader(
            "./testoutput/tiled_tiff_dataset_inverse/color/1/006/001/002.tif"
        ).read(),
    )


def test_view_write_without_open() -> None:
    # This test would be the same for TiffDataset

    delete_dir("./testoutput/wk_dataset_write_without_open")

    ds = WKDataset.create("./testoutput/wk_dataset_write_without_open", scale=(1, 1, 1))
    ds.add_layer("color", Layer.COLOR_TYPE)

    ds.get_layer("color").add_mag("1")

    wk_view = ds.get_view("color", "1", size=(32, 64, 16), is_bounded=False)

    assert not wk_view._is_opened

    write_data = (np.random.rand(32, 64, 16) * 255).astype(np.uint8)
    wk_view.write(write_data)

    assert not wk_view._is_opened


def test_typing_of_get_mag() -> None:
    ds = WKDataset("./testdata/simple_wk_dataset")
    layer = ds.get_layer("color")
    assert layer.get_mag("1") == layer.get_mag(1)
    assert layer.get_mag("1") == layer.get_mag((1, 1, 1))
    assert layer.get_mag("1") == layer.get_mag([1, 1, 1])
    assert layer.get_mag("1") == layer.get_mag(np.array([1, 1, 1]))
    assert layer.get_mag("1") == layer.get_mag(Mag(1))


def test_wk_dataset_get_or_create() -> None:
    delete_dir("./testoutput/wk_dataset_get_or_create")

    # dataset does not exists yet
    ds1 = WKDataset.get_or_create(
        "./testoutput/wk_dataset_get_or_create", scale=(1, 1, 1)
    )
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", Layer.COLOR_TYPE)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = WKDataset.get_or_create(
        "./testoutput/wk_dataset_get_or_create", scale=(1, 1, 1)
    )
    assert "color" in ds2.layers.keys()

    try:
        # dataset already exists, but with a different scale
        WKDataset.get_or_create(
            "./testoutput/wk_dataset_get_or_create", scale=(2, 2, 2)
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass


def test_tiff_dataset_get_or_create() -> None:
    delete_dir("./testoutput/tiff_dataset_get_or_create")

    # dataset does not exists yet
    ds1 = TiffDataset.get_or_create(
        "./testoutput/tiff_dataset_get_or_create", scale=(1, 1, 1)
    )
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", Layer.COLOR_TYPE)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = TiffDataset.get_or_create(
        "./testoutput/tiff_dataset_get_or_create", scale=(1, 1, 1)
    )
    assert "color" in ds2.layers.keys()

    try:
        # dataset already exists, but with a different scale
        TiffDataset.get_or_create(
            "./testoutput/tiff_dataset_get_or_create", scale=(2, 2, 2)
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass

    try:
        # dataset already exists, but with a different pattern
        TiffDataset.get_or_create(
            "./testoutput/tiff_dataset_get_or_create",
            scale=(1, 1, 1),
            pattern="ds_{zzz}.tif",
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass


def test_tiled_tiff_dataset_get_or_create() -> None:
    delete_dir("./testoutput/tiled_tiff_dataset_get_or_create")

    # dataset does not exists yet
    ds1 = TiledTiffDataset.get_or_create(
        "./testoutput/tiled_tiff_dataset_get_or_create",
        scale=(1, 1, 1),
        tile_size=(32, 64),
    )
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", Layer.COLOR_TYPE)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = TiledTiffDataset.get_or_create(
        "./testoutput/tiled_tiff_dataset_get_or_create",
        scale=(1, 1, 1),
        tile_size=(32, 64),
    )
    assert "color" in ds2.layers.keys()

    try:
        # dataset already exists, but with a different scale
        TiledTiffDataset.get_or_create(
            "./testoutput/tiled_tiff_dataset_get_or_create",
            scale=(2, 2, 2),
            tile_size=(32, 64),
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass

    try:
        # dataset already exists, but with a different tile_size
        TiledTiffDataset.get_or_create(
            "./testoutput/tiled_tiff_dataset_get_or_create",
            scale=(1, 1, 1),
            tile_size=(100, 100),
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass

    try:
        # dataset already exists, but with a different pattern
        TiledTiffDataset.get_or_create(
            "./testoutput/tiled_tiff_dataset_get_or_create",
            scale=(1, 1, 1),
            tile_size=(32, 64),
            pattern="ds_{zzz}.tif",
        )
        raise Exception(expected_error_msg)
    except AssertionError:
        pass


def test_changing_layer_bounding_box() -> None:
    delete_dir("./testoutput/test_changing_layer_bounding_box/")
    copytree(
        "./testdata/simple_tiff_dataset/",
        "./testoutput/test_changing_layer_bounding_box/",
    )

    ds = TiffDataset("./testoutput/test_changing_layer_bounding_box/")
    layer = ds.get_layer("color")
    mag = layer.get_mag("1")

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (265, 265, 10)
    original_data = mag.read(size=bbox_size)
    assert original_data.shape == (1, 265, 265, 10)

    layer.set_bounding_box_size((100, 100, 10))  # decrease boundingbox

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (100, 100, 10)
    less_data = mag.read(size=bbox_size)
    assert less_data.shape == (1, 100, 100, 10)
    assert np.array_equal(original_data[:, :100, :100, :10], less_data)

    layer.set_bounding_box_size((300, 300, 10))  # increase the boundingbox

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (300, 300, 10)
    more_data = mag.read(size=bbox_size)
    assert more_data.shape == (1, 300, 300, 10)
    assert np.array_equal(more_data[:, :265, :265, :10], original_data)

    layer.set_bounding_box_size((300, 300, 10))  # increase the boundingbox

    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)

    layer.set_bounding_box(
        offset=(10, 10, 0), size=(255, 255, 10)
    )  # change offset and size

    new_bbox_offset = ds.properties.data_layers["color"].get_bounding_box_offset()
    new_bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert new_bbox_offset == (10, 10, 0)
    assert new_bbox_size == (255, 255, 10)
    new_data = mag.read(size=new_bbox_size)
    assert new_data.shape == (1, 255, 255, 10)
    assert np.array_equal(original_data[:, 10:, 10:, :], new_data)


def test_view_offsets() -> None:
    delete_dir("./testoutput/wk_offset_tests")

    ds = WKDataset.create("./testoutput/wk_offset_tests", scale=(1, 1, 1))
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")

    # The dataset is new -> no data has been written.
    # Therefore, the size of the bounding box in the properties.json is (0, 0, 0)

    # Creating this view works because the size is set to (0, 0, 0)
    # However, in practice such a view would not make sense because 'is_bounded' is set to 'True'
    wk_view = ds.get_view("color", "1", size=(0, 0, 0), is_bounded=True)
    assert wk_view.global_offset == tuple((0, 0, 0))
    assert wk_view.size == tuple((0, 0, 0))

    try:
        # Creating this view does not work because the size (16, 16, 16) would exceed the boundingbox from the properties.json
        ds.get_view("color", "1", size=(16, 16, 16), is_bounded=True)
        raise Exception(expected_error_msg)
    except AssertionError:
        pass

    # This works because 'is_bounded' is set to 'False'
    # Therefore, the bounding box of the view can be larger than the bounding box from the properties.json
    wk_view = ds.get_view("color", "1", size=(16, 16, 16), is_bounded=False)
    assert wk_view.global_offset == tuple((0, 0, 0))
    assert wk_view.size == tuple((16, 16, 16))

    np.random.seed(1234)
    write_data = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    mag.write(write_data, offset=(10, 20, 30))

    # The bounding box of the dataset was updated according to the written data
    # Therefore, creating a view with a size of (16, 16, 16) is now allowed
    wk_view = ds.get_view("color", "1", size=(16, 16, 16), is_bounded=True)
    assert wk_view.global_offset == tuple((10, 20, 30))
    assert wk_view.size == tuple((16, 16, 16))

    try:
        # Creating this view does not work because the offset (0, 0, 0) would be outside of the boundingbox from the properties.json
        ds.get_view("color", "1", size=(16, 16, 16), offset=(0, 0, 0), is_bounded=True)
        raise Exception(expected_error_msg)
    except AssertionError:
        pass

    # Creating this view works, even though the offset (0, 0, 0) is outside of the boundingbox from the properties.json, because 'is_bounded' is set to 'False'
    wk_view = ds.get_view(
        "color", "1", size=(16, 16, 16), offset=(0, 0, 0), is_bounded=False
    )
    assert wk_view.global_offset == tuple((0, 0, 0))
    assert wk_view.size == tuple((16, 16, 16))

    # Creating this view works because the bounding box of the view is inside the bounding box from the properties.json
    wk_view = ds.get_view(
        "color", "1", size=(16, 16, 16), offset=(20, 30, 40), is_bounded=True
    )
    assert wk_view.global_offset == tuple((20, 30, 40))
    assert wk_view.size == tuple((16, 16, 16))

    # Creating this subview works because the subview is completely inside the 'wk_view'
    sub_view = wk_view.get_view(size=(8, 8, 8), relative_offset=(8, 8, 8))
    assert sub_view.global_offset == tuple((28, 38, 48))
    assert sub_view.size == tuple((8, 8, 8))

    try:
        # Creating this subview does not work because it is not completely inside the 'wk_view'
        wk_view.get_view(size=(10, 10, 10), relative_offset=(8, 8, 8))
        raise Exception(expected_error_msg)
    except AssertionError:
        pass


def test_adding_layer_with_invalid_dtype_per_layer() -> None:
    delete_dir("./testoutput/invalid_dtype")

    ds = WKDataset.create("./testoutput/invalid_dtype", scale=(1, 1, 1))
    with pytest.raises(TypeError):
        # this would lead to a dtype_per_channel of "uint10", but that is not a valid dtype
        ds.add_layer(
            "color", Layer.COLOR_TYPE, dtype_per_layer="uint30", num_channels=3
        )
    with pytest.raises(TypeError):
        # "int" is interpreted as "int64", but 64 bit cannot be split into 3 channels
        ds.add_layer("color", Layer.COLOR_TYPE, dtype_per_layer="int", num_channels=3)
    ds.add_layer(
        "color", Layer.COLOR_TYPE, dtype_per_layer="int", num_channels=4
    )  # "int"/"int64" works with 4 channels


def test_adding_layer_with_valid_dtype_per_layer() -> None:
    delete_dir("./testoutput/valid_dtype")

    ds = WKDataset.create("./testoutput/valid_dtype", scale=(1, 1, 1))
    ds.add_layer("color1", Layer.COLOR_TYPE, dtype_per_layer="uint24", num_channels=3)
    ds.add_layer("color2", Layer.COLOR_TYPE, dtype_per_layer=np.uint8, num_channels=1)
    ds.add_layer("color3", Layer.COLOR_TYPE, dtype_per_channel=np.uint8, num_channels=3)
    ds.add_layer("color4", Layer.COLOR_TYPE, dtype_per_channel="uint8", num_channels=3)


def test_writing_subset_of_compressed_data_multi_channel() -> None:
    delete_dir("./testoutput/compressed_data/")

    # create uncompressed dataset
    write_data1 = (np.random.rand(3, 20, 40, 60) * 255).astype(np.uint8)
    WKDataset.create(
        os.path.abspath("./testoutput/compressed_data"), scale=(1, 1, 1)
    ).add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag(
        "1", block_len=8, file_len=8
    ).write(
        write_data1
    )

    # compress data
    compress_mag_inplace(
        os.path.abspath("./testoutput/compressed_data/"),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        WKDataset("./testoutput/compressed_data").get_layer("color").get_mag("1")
    )

    write_data2 = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    compressed_mag.write(
        offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
    )

    np.array_equal(
        write_data2, compressed_mag.read(offset=(10, 20, 30), size=(10, 10, 10))
    )  # the new data was written
    np.array_equal(
        write_data1[:, :10, :20, :30],
        compressed_mag.read(offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


def test_writing_subset_of_compressed_data_single_channel() -> None:
    delete_dir("./testoutput/compressed_data/")

    # create uncompressed dataset
    write_data1 = (np.random.rand(20, 40, 60) * 255).astype(np.uint8)
    WKDataset.create(
        os.path.abspath("./testoutput/compressed_data"), scale=(1, 1, 1)
    ).add_layer("color", Layer.COLOR_TYPE).add_mag("1", block_len=8, file_len=8).write(
        write_data1
    )

    # compress data
    compress_mag_inplace(
        os.path.abspath("./testoutput/compressed_data/"),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        WKDataset("./testoutput/compressed_data").get_layer("color").get_mag("1")
    )

    write_data2 = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    compressed_mag.write(
        offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
    )

    np.array_equal(
        write_data2, compressed_mag.read(offset=(10, 20, 30), size=(10, 10, 10))
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_mag.read(offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


def test_writing_subset_of_compressed_data() -> None:
    delete_dir("./testoutput/compressed_data/")

    # create uncompressed dataset
    WKDataset.create(
        os.path.abspath("./testoutput/compressed_data"), scale=(1, 1, 1)
    ).add_layer("color", Layer.COLOR_TYPE).add_mag("1", block_len=8, file_len=8).write(
        (np.random.rand(20, 40, 60) * 255).astype(np.uint8)
    )

    # compress data
    compress_mag_inplace(
        os.path.abspath("./testoutput/compressed_data/"),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        WKDataset("./testoutput/compressed_data").get_layer("color").get_mag("1")
    )

    with pytest.raises(WKWException):
        # calling 'write' with unaligned data on compressed data without setting 'allow_compressed_write=True'
        compressed_mag.write(
            offset=(10, 20, 30),
            data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
        )


def test_writing_subset_of_chunked_compressed_data() -> None:
    delete_dir("./testoutput/compressed_data/")

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    WKDataset.create(
        os.path.abspath("./testoutput/compressed_data"), scale=(1, 1, 1)
    ).add_layer("color", Layer.COLOR_TYPE).add_mag("1", block_len=8, file_len=8).write(
        write_data1
    )

    # compress data
    compress_mag_inplace(
        os.path.abspath("./testoutput/compressed_data/"),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_view = WKDataset("./testoutput/compressed_data").get_view(
        "color", "1", size=(100, 200, 300), is_bounded=True
    )

    with pytest.raises(AssertionError):
        # the aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
        compressed_view.write(
            relative_offset=(10, 20, 30),
            data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
            allow_compressed_write=True,
        )

    # the aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
    write_data2 = (np.random.rand(50, 40, 30) * 255).astype(np.uint8)
    compressed_view.write(
        relative_offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
    )

    np.array_equal(
        write_data2, compressed_view.read(offset=(10, 20, 30), size=(50, 40, 30))
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_view.read(offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


def test_add_symlink_layer() -> None:
    delete_dir("./testoutput/wk_dataset_with_symlink")
    delete_dir("./testoutput/simple_wk_dataset_copy")
    copytree("./testdata/simple_wk_dataset/", "./testoutput/simple_wk_dataset_copy/")

    original_mag = (
        WKDataset("./testoutput/simple_wk_dataset_copy/")
        .get_layer("color")
        .get_mag("1")
    )

    ds = WKDataset.create("./testoutput/wk_dataset_with_symlink", scale=(1, 1, 1))
    symlink_layer = ds.add_symlink_layer("./testoutput/simple_wk_dataset_copy/color/")
    mag = symlink_layer.get_mag("1")

    assert path.exists("./testoutput/wk_dataset_with_symlink/color/1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1

    # write data in symlink layer
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data)

    assert np.array_equal(mag.read(size=(10, 10, 10)), write_data)
    assert np.array_equal(original_mag.read(size=(10, 10, 10)), write_data)


def test_search_dataset_also_for_long_layer_name() -> None:
    delete_dir("./testoutput/long_layer_name")

    ds = WKDataset.create("./testoutput/long_layer_name", scale=(1, 1, 1))
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("2")

    assert mag.name == "2"
    short_mag_file_path = join(ds.path, "color", Mag(mag.name).to_layer_name())
    long_mag_file_path = join(ds.path, "color", Mag(mag.name).to_long_layer_name())

    assert os.path.exists(short_mag_file_path)
    assert not os.path.exists(long_mag_file_path)

    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data, offset=(10, 10, 10))

    assert np.array_equal(
        mag.read(offset=(10, 10, 10), size=(10, 10, 10)), np.expand_dims(write_data, 0)
    )

    # rename the path from "long_layer_name/color/2" to "long_layer_name/color/2-2-2"
    os.rename(short_mag_file_path, long_mag_file_path)

    with pytest.raises(WKWException):
        # the dataset has to be reopened to notice the changed directory
        mag.read(offset=(10, 10, 10), size=(10, 10, 10))

    # when opening the dataset, it searches both for the long and the short path
    layer = WKDataset("./testoutput/long_layer_name").get_layer("color")
    mag = layer.get_mag("2")
    assert np.array_equal(
        mag.read(offset=(10, 10, 10), size=(10, 10, 10)), np.expand_dims(write_data, 0)
    )
    layer.delete_mag("2")


def test_outdated_dtype_parameter() -> None:
    delete_dir("./testoutput/outdated_dtype")

    ds = WKDataset.create("./testoutput/outdated_dtype", scale=(1, 1, 1))
    with pytest.raises(ValueError):
        ds.get_or_add_layer("color", Layer.COLOR_TYPE, dtype=np.uint8, num_channels=1)

    with pytest.raises(ValueError):
        ds.add_layer("color", Layer.COLOR_TYPE, dtype=np.uint8, num_channels=1)


def test_dataset_conversion() -> None:
    origin_wk_ds_path = "./testoutput/conversion/origin_wk/"
    origin_tiff_ds_path = "./testoutput/conversion/origin_tiff/"

    wk_to_tiff_ds_path = "./testoutput/conversion/wk_to_tiff/"
    tiff_to_wk_ds_path = "./testoutput/conversion/tiff_to_wk/"

    wk_to_tiff_to_wk_ds_path = "./testoutput/conversion/wk_to_tiff_to_wk/"
    tiff_to_wk_to_tiff_ds_path = "./testoutput/conversion/tiff_to_wk_to_tiff/"

    delete_dir(origin_wk_ds_path)
    delete_dir(origin_tiff_ds_path)
    delete_dir(wk_to_tiff_ds_path)
    delete_dir(tiff_to_wk_ds_path)
    delete_dir(wk_to_tiff_to_wk_ds_path)
    delete_dir(tiff_to_wk_to_tiff_ds_path)

    # create example dataset
    origin_wk_ds = WKDataset.create(origin_wk_ds_path, scale=(1, 1, 1))
    wk_seg_layer = origin_wk_ds.add_layer(
        "layer1", Layer.SEGMENTATION_TYPE, num_channels=1, largest_segment_id=1000000000
    )
    wk_seg_layer.add_mag("1", block_len=8, file_len=16).write(
        offset=(10, 20, 30), data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8)
    )
    wk_seg_layer.add_mag("2", block_len=8, file_len=16).write(
        offset=(5, 10, 15), data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8)
    )
    wk_color_layer = origin_wk_ds.add_layer("layer2", Layer.COLOR_TYPE, num_channels=3)
    wk_color_layer.add_mag("1", block_len=8, file_len=16).write(
        offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
    )
    wk_color_layer.add_mag("2", block_len=8, file_len=16).write(
        offset=(5, 10, 15), data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8)
    )
    wk_to_tiff_ds = origin_wk_ds.to_tiff_dataset(wk_to_tiff_ds_path)
    wk_to_tiff_to_wk_ds = wk_to_tiff_ds.to_wk_dataset(wk_to_tiff_to_wk_ds_path)

    assert origin_wk_ds.layers.keys() == wk_to_tiff_to_wk_ds.layers.keys()
    for layer_name in origin_wk_ds.layers:
        assert (
            origin_wk_ds.layers[layer_name].mags.keys()
            == wk_to_tiff_to_wk_ds.layers[layer_name].mags.keys()
        )
        for mag in origin_wk_ds.layers[layer_name].mags:
            origin_header = origin_wk_ds.layers[layer_name].mags[mag].header
            converted_header = wk_to_tiff_to_wk_ds.layers[layer_name].mags[mag].header
            assert origin_header.voxel_type == converted_header.voxel_type
            assert origin_header.num_channels == converted_header.num_channels
            assert origin_header.block_type == converted_header.block_type
            # the block_length and file_length might differ because the conversion from tiff to wk uses the defaults
            assert np.array_equal(
                origin_wk_ds.layers[layer_name].mags[mag].read(),
                wk_to_tiff_to_wk_ds.layers[layer_name].mags[mag].read(),
            )

    # create example dataset
    origin_tiff_ds = TiffDataset.create(
        origin_tiff_ds_path, scale=(1, 1, 1), pattern="z_dim_{zzzzz}.tif"
    )
    tiff_seg_layer = origin_tiff_ds.add_layer(
        "layer1", Layer.SEGMENTATION_TYPE, num_channels=1, largest_segment_id=1000000000
    )
    tiff_seg_layer.add_mag("1").write(
        offset=(10, 20, 30), data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8)
    )
    tiff_seg_layer.add_mag("2").write(
        offset=(5, 10, 15), data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8)
    )
    tiff_color_layer = origin_tiff_ds.add_layer(
        "layer2", Layer.COLOR_TYPE, num_channels=3
    )
    tiff_color_layer.add_mag("1").write(
        offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
    )
    tiff_color_layer.add_mag("2").write(
        offset=(5, 10, 15), data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8)
    )

    tiff_to_wk_ds = origin_tiff_ds.to_wk_dataset(tiff_to_wk_ds_path)
    tiff_to_wk_to_tiff = tiff_to_wk_ds.to_tiff_dataset(
        tiff_to_wk_to_tiff_ds_path, pattern="different_pattern_{zzzzz}.tif"
    )

    assert origin_tiff_ds.layers.keys() == tiff_to_wk_to_tiff.layers.keys()
    for layer_name in origin_tiff_ds.layers:
        assert (
            origin_tiff_ds.layers[layer_name].mags.keys()
            == tiff_to_wk_to_tiff.layers[layer_name].mags.keys()
        )
        for mag in origin_tiff_ds.layers[layer_name].mags:
            origin_header = origin_tiff_ds.layers[layer_name].mags[mag].header
            converted_header = tiff_to_wk_to_tiff.layers[layer_name].mags[mag].header
            assert origin_header.dtype_per_channel == converted_header.dtype_per_channel
            assert origin_header.num_channels == converted_header.num_channels
            assert origin_header.tile_size == converted_header.tile_size
            # the pattern of the datasets does not match because I intentionally used two different paths in this test
            assert np.array_equal(
                origin_tiff_ds.layers[layer_name].mags[mag].read(),
                tiff_to_wk_to_tiff.layers[layer_name].mags[mag].read(),
            )


def test_for_zipped_chunks() -> None:
    delete_dir("./testoutput/zipped_chunking_source/")
    delete_dir("./testoutput/zipped_chunking_target/")
    copytree("./testdata/simple_wk_dataset/", "./testoutput/zipped_chunking_source/")

    source_view = WKDataset("./testoutput/zipped_chunking_source/").get_view(
        "color", "1", size=(256, 256, 256), is_bounded=False
    )

    target_mag = (
        WKDataset.create("./testoutput/zipped_chunking_target/", scale=(1, 1, 1))
        .get_or_add_layer(
            "color", Layer.COLOR_TYPE, dtype_per_channel="uint8", num_channels=3
        )
        .get_or_add_mag("1", block_len=8, file_len=4)
    )

    target_mag.layer.dataset.properties._set_bounding_box_of_layer(
        "color", offset=(0, 0, 0), size=(256, 256, 256)
    )
    target_view = target_mag.get_view(size=(256, 256, 256))

    with get_executor_for_args(None) as executor:
        func = named_partial(
            copy_and_transform_job, name="foo", val=42
        )  # curry the function with further arguments
        source_view.for_zipped_chunks(
            func,
            target_view=target_view,
            source_chunk_size=(64, 64, 64),  # multiple of (wkw_file_len,) * 3
            target_chunk_size=(64, 64, 64),  # multiple of (wkw_file_len,) * 3
            executor=executor,
        )

    assert np.array_equal(
        source_view.read(size=source_view.size) + 50,
        target_view.read(size=target_view.size),
    )


def test_for_zipped_chunks_invalid_target_chunk_size_wk() -> None:
    delete_dir("./testoutput/zipped_chunking_source_invalid/")

    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    layer = WKDataset.create(
        "./testoutput/zipped_chunking_source_invalid/", scale=(1, 1, 1)
    ).get_or_add_layer("color", Layer.COLOR_TYPE)
    source_mag_dataset = layer.get_or_add_mag(1, block_len=8, file_len=8)
    target_mag_dataset = layer.get_or_add_mag(2, block_len=8, file_len=8)
    source_mag_dataset.write(
        data=(np.random.rand(1, 300, 300, 300) * 255).astype(np.uint8)
    )
    source_view = source_mag_dataset.get_view()
    target_view = target_mag_dataset.get_view(size=source_view.size, is_bounded=False)

    def func(args: Tuple[View, View, int]) -> None:
        (s, t, i) = args

    with get_executor_for_args(None) as executor:
        for test_case in test_cases_wk:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    work_on_chunk=func,
                    target_view=target_view,
                    source_chunk_size=test_case,
                    target_chunk_size=test_case,
                    executor=executor,
                )


def test_for_zipped_chunks_invalid_target_chunk_size_tiled_tiff() -> None:
    delete_dir("./testoutput/zipped_chunking_source_invalid/")

    test_cases = [
        (10, 20, 10),
        (64, 50, 5),
        (200, 128, 12),
    ]

    layer = TiledTiffDataset.create(
        "./testoutput/zipped_chunking_source_invalid/",
        scale=(1, 1, 1),
        tile_size=(64, 64),
    ).get_or_add_layer("color", Layer.COLOR_TYPE)
    source_mag_dataset = layer.get_or_add_mag(1)
    target_mag_dataset = layer.get_or_add_mag(2)
    source_mag_dataset.write(
        data=(np.random.rand(1, 300, 300, 10) * 255).astype(np.uint8)
    )
    source_view = source_mag_dataset.get_view()
    target_view = target_mag_dataset.get_view(size=source_view.size, is_bounded=False)

    def func(args: Tuple[View, View, int]) -> None:
        (s, t, i) = args

    with get_executor_for_args(None) as executor:
        for test_case in test_cases:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    work_on_chunk=func,
                    target_view=target_view,
                    source_chunk_size=test_case,
                    target_chunk_size=test_case,
                    executor=executor,
                )


def test_for_zipped_chunks_invalid_target_chunk_size_tiff() -> None:

    test_cases = [  # offset, size, chunk_size
        ((0, 0, 0), (64, 64, 10), (32, 64, 5)),
        ((14, 14, 5), (46, 46, 5), (32, 32, 5)),
    ]

    def func(args: Tuple[View, View, int]) -> None:
        (s, t, i) = args

    for offset, size, chunk_size in test_cases:
        delete_dir("./testoutput/zipped_chunking_source_invalid/")

        ds = TiffDataset.create(
            "./testoutput/zipped_chunking_source_invalid/", scale=(1, 1, 1)
        )
        color_layer = ds.get_or_add_layer("color", Layer.COLOR_TYPE)
        seg_layer = ds.get_or_add_layer(
            "seg", Layer.SEGMENTATION_TYPE, largest_segment_id=10000000
        )
        source_mag_dataset = color_layer.get_or_add_mag(1)
        target_mag_dataset = seg_layer.get_or_add_mag(1)
        source_mag_dataset.write(
            data=(np.random.rand(1, *size) * 255).astype(np.uint8), offset=offset
        )
        source_view = source_mag_dataset.get_view()
        target_view = target_mag_dataset.get_view(
            size=source_view.size, is_bounded=False
        )

        with get_executor_for_args(None) as executor:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    work_on_chunk=func,
                    target_view=target_view,
                    source_chunk_size=chunk_size,
                    target_chunk_size=chunk_size,
                    executor=executor,
                )


def test_read_only_view() -> None:
    delete_dir("./testoutput/read_only_view/")
    ds = WKDataset.create("./testoutput/read_only_view/", scale=(1, 1, 1))
    mag = ds.get_or_add_layer("color", Layer.COLOR_TYPE).get_or_add_mag("1")
    mag.write(
        data=(np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8), offset=(10, 20, 30)
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = (np.random.rand(1, 5, 6, 7) * 255).astype(np.uint8)
    with pytest.raises(AssertionError):
        v_read.write(data=new_data)

    v_write.write(data=new_data)


@pytest.fixture(
    params=[
        WKDataset,
        TiffDataset,
        TiledTiffDataset,
    ]
)
def create_dataset(request: Any) -> Generator[MagDataset, None, None]:
    dataset_type = request.param
    with tempfile.TemporaryDirectory() as temp_dir:
        if dataset_type == TiledTiffDataset:
            ds = dataset_type.create(temp_dir, scale=(2, 2, 1), tile_size=(64, 32))
        else:
            ds = dataset_type.create(temp_dir, scale=(2, 2, 1))

        if dataset_type == WKDataset:
            mag = ds.add_layer("color", "color").add_mag(
                "2-2-1", block_len=8, file_len=8
            )  # cube_size = 8*8 = 64
        else:
            mag = ds.add_layer("color", "color").add_mag("2-2-1")
        yield mag


def test_bounding_box_on_disk(create_dataset: MagDataset) -> None:
    mag = create_dataset

    write_positions = [(0, 0, 0), (20, 80, 120), (1000, 2000, 4000)]
    data_size = (10, 20, 30)
    write_data = (np.random.rand(*data_size) * 255).astype(np.uint8)
    for offset in write_positions:
        mag.write(offset=offset, data=write_data)

    bounding_boxes_on_disk = list(mag.get_bounding_boxes_on_disk())
    file_size = mag._get_file_dimensions()

    expected_results = set()
    for offset in write_positions:
        # enumerate all bounding boxes of the current write operation
        x_range = range(
            offset[0] // file_size[0] * file_size[0],
            offset[0] + data_size[0],
            file_size[0],
        )
        y_range = range(
            offset[1] // file_size[1] * file_size[1],
            offset[1] + data_size[1],
            file_size[1],
        )
        z_range = range(
            offset[2] // file_size[2] * file_size[2],
            offset[2] + data_size[2],
            file_size[2],
        )

        for bb_offset in itertools.product(x_range, y_range, z_range):
            expected_results.add((bb_offset, file_size))

    assert set(bounding_boxes_on_disk) == expected_results


def test_compression() -> None:
    temp_ds_path = Path("testoutput") / "compressed_ds"
    delete_dir(str(temp_ds_path))
    copytree(Path("testdata", "simple_wk_dataset"), temp_ds_path)

    mag1 = WKDataset(temp_ds_path).get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data)

    mag1.compress()

    assert np.array_equal(write_data, mag1.read(size=(10, 20, 30)))

    with pytest.raises(wkw.WKWException):
        # writing unaligned data to a compressed dataset
        mag1.write((np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8))

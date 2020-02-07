import filecmp
import json
from os.path import dirname

import numpy as np
from shutil import rmtree, copytree

from wkcuber.api.Dataset import WKDataset, TiffDataset, TiledTiffDataset
from os import path, makedirs

from wkcuber.api.Layer import Layer
from wkcuber.api.Properties import TiffProperties, WKProperties
from wkcuber.api.TiffData.TiffMag import TiffReader
from wkcuber.mag import Mag


def delete_dir(relative_path):
    if path.exists(relative_path) and path.isdir(relative_path):
        rmtree(relative_path)


def test_create_wk_dataset_with_layer_and_mag():
    delete_dir("./testoutput/wk_dataset")

    ds = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("./testoutput/wk_dataset/color/1")
    assert path.exists("./testoutput/wk_dataset/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2


def test_create_wk_dataset_with_explicit_header_fields():
    delete_dir("./testoutput/wk_dataset_advanced")

    ds = WKDataset.create("./testoutput/wk_dataset_advanced", scale=(1, 1, 1))
    ds.add_layer("color", "color", dtype=np.uint16, num_channels=3)

    ds.get_layer("color").add_mag("1", block_len=64, file_len=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("./testoutput/wk_dataset_advanced/color/1")
    assert path.exists("./testoutput/wk_dataset_advanced/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2

    assert ds.properties.data_layers["color"].element_class == np.dtype(np.uint16)
    assert (
        ds.properties.data_layers["color"].wkw_magnifications[0].cube_length == 64 * 64
    )  # mag "1"
    assert ds.properties.data_layers["color"].wkw_magnifications[0].mag == Mag("1")
    assert (
        ds.properties.data_layers["color"].wkw_magnifications[1].cube_length == 32 * 32
    )  # mag "2-2-1" (defaults are used)
    assert ds.properties.data_layers["color"].wkw_magnifications[1].mag == Mag("2-2-1")


def test_create_tiff_dataset_with_layer_and_mag():
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


def test_open_wk_dataset():
    ds = WKDataset("./testdata/simple_wk_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1


def test_open_tiff_dataset():
    ds = TiffDataset("./testdata/simple_tiff_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1


def test_view_read_with_open():
    # This test would be the same for TiffDataset

    wk_view = WKDataset("./testdata/simple_wk_dataset/").get_view(
        "color", "1", size=(32, 32, 32)
    )

    assert not wk_view._is_opened

    with wk_view.open():
        assert wk_view._is_opened

        data = wk_view.read((10, 10, 10))
        assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_view_read_without_open():
    # This test would be the same for TiffDataset

    wk_view = WKDataset("./testdata/simple_wk_dataset/").get_view(
        "color", "1", size=(32, 32, 32)
    )

    assert not wk_view._is_opened

    # 'read()' checks if it was already opened. If not, it opens and closes automatically
    data = wk_view.read((10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_view_wk_write():
    delete_dir("./testoutput/simple_wk_dataset/")
    copytree("./testdata/simple_wk_dataset/", "./testoutput/simple_wk_dataset/")

    wk_view = WKDataset("./testoutput/simple_wk_dataset/").get_view(
        "color", "1", size=(100, 100, 100)
    )

    with wk_view.open():
        np.random.seed(1234)
        write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

        wk_view.write(write_data)

        data = wk_view.read((10, 10, 10))
        assert np.array_equal(data, write_data)


def test_view_tiff_write():
    delete_dir("./testoutput/simple_tiff_dataset/")
    copytree("./testdata/simple_tiff_dataset/", "./testoutput/simple_tiff_dataset/")

    tiff_view = TiffDataset("./testoutput/simple_tiff_dataset/").get_view(
        "color", "1", size=(100, 100, 100)
    )

    with tiff_view.open():
        np.random.seed(1234)
        write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)

        tiff_view.write(write_data)

        data = tiff_view.read((5, 5, 5))
        assert data.shape == (1, 5, 5, 5)  # this dataset has only one channel
        assert np.array_equal(data, np.expand_dims(write_data, 0))


def test_view_tiff_write_out_of_bounds():
    new_dataset_path = "./testoutput/tiff_view_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_tiff_dataset/", new_dataset_path)

    tiff_view = TiffDataset(new_dataset_path).get_view(
        "color", "1", size=(100, 100, 100)
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


def test_view_wk_write_out_of_bounds():
    new_dataset_path = "./testoutput/wk_view_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("./testdata/simple_wk_dataset/", new_dataset_path)

    tiff_view = WKDataset(new_dataset_path).get_view("color", "1", size=(100, 100, 100))

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


def test_tiff_write_out_of_bounds():
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


def test_wk_write_out_of_bounds():
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


def test_update_new_bounding_box_offset():
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


def test_other_file_extensions_for_tiff_dataset():
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
    assert np.array_equal(mag.read((10, 10, 10)), np.expand_dims(write_data, 0))


def test_tiff_write_multi_channel_uint8():
    dataset_path = "./testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint8
    data = np.zeros((3, 250, 250, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i
                data[1, i, j, h] = j
                data[2, i, j, h] = 100

    ds_tiff.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 250, 10)))


def test_wk_write_multi_channel_uint8():
    dataset_path = "./testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", Layer.COLOR_TYPE, num_channels=3).add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint8
    data = np.zeros((3, 250, 250, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i
                data[1, i, j, h] = j
                data[2, i, j, h] = 100

    ds_tiff.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 250, 10)))


def test_tiff_write_multi_channel_uint16():
    dataset_path = "./testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer(
        "color", Layer.COLOR_TYPE, num_channels=3, dtype=np.uint16
    ).add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint16
    data = np.zeros((3, 250, 250, 10), dtype=np.uint16)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256

    mag.write(data)
    written_data = mag.read(size=(250, 250, 10))

    print(written_data.dtype)

    assert np.array_equal(data, written_data)


def test_wk_write_multi_channel_uint16():
    dataset_path = "./testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer(
        "color", Layer.COLOR_TYPE, num_channels=3, dtype=np.uint16
    ).add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint16
    data = np.zeros((3, 250, 250, 10), dtype=np.uint16)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256

    mag.write(data)
    written_data = mag.read(size=(250, 250, 10))

    print(written_data.dtype)

    assert np.array_equal(data, written_data)


def test_wkw_empty_read():
    filename = "./testoutput/empty_wk_dataset"
    delete_dir(filename)

    mag = (
        WKDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE)
        .add_mag("1")
    )
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_empty_read():
    filename = "./testoutput/empty_tiff_dataset"
    delete_dir(filename)

    mag = (
        TiffDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", Layer.COLOR_TYPE)
        .add_mag("1")
    )
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_read_padded_data():
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


def test_wk_read_padded_data():
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


def test_read_and_write_of_properties():
    destination_path = "./testoutput/read_write_properties/"
    delete_dir(destination_path)
    source_file_name = "./testdata/simple_tiff_dataset/datasource-properties.json"
    destination_file_name = destination_path + "datasource-properties.json"

    imported_properties = TiffProperties._from_json(source_file_name)
    imported_properties._path = destination_file_name
    makedirs(destination_path)
    imported_properties._export_as_json()

    filecmp.cmp(source_file_name, destination_file_name)


def test_num_channel_mismatch_assertion():
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


def test_get_or_add_layer():
    # This test would be the same for TiffDataset

    delete_dir("./testoutput/wk_dataset")

    ds = WKDataset.create("./testoutput/wk_dataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color", Layer.COLOR_TYPE, dtype=np.uint8, num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color", Layer.COLOR_TYPE, dtype=np.uint8, num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    try:
        # layer did exist before but with another 'dtype' (this would work the same for 'category' and 'num_channels')
        layer = ds.get_or_add_layer(
            "color", Layer.COLOR_TYPE, dtype=np.uint16, num_channels=1
        )

        raise Exception(
            "The test 'test_get_or_add_layer' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_mag_for_wk():
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


def test_get_or_add_mag_for_tiff():
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


def test_tiled_tiff_read_and_write_multichannel():
    delete_dir("./testoutput/TiledTiffDataset")
    tiled_tiff_ds = TiledTiffDataset.create(
        "./testoutput/TiledTiffDataset",
        scale=(1, 1, 1),
        tile_size=(32, 32),
        pattern="{xxx}_{yyy}_{zzz}.tif",
    )

    mag = tiled_tiff_ds.add_layer("color", "color", num_channels=3).add_mag("1")

    data = np.zeros((3, 250, 250, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i
                data[1, i, j, h] = j
                data[2, i, j, h] = 100

    mag.write(data, offset=(5, 5, 5))
    written_data = mag.read(size=(250, 250, 10), offset=(5, 5, 5))
    assert written_data.shape == (3, 250, 250, 10)
    assert np.array_equal(data, written_data)


def test_tiled_tiff_read_and_write():
    delete_dir("./testoutput/tiled_tiff_dataset")
    tiled_tiff_ds = TiledTiffDataset.create(
        "./testoutput/tiled_tiff_dataset",
        scale=(1, 1, 1),
        tile_size=(32, 32),
        pattern="{xxx}_{yyy}_{zzz}.tif",
    )

    mag = tiled_tiff_ds.add_layer("color", "color").add_mag("1")

    data = np.zeros((250, 250, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[i, j, h] = i + j % 250

    mag.write(data, offset=(5, 5, 5))
    written_data = mag.read(size=(250, 250, 10), offset=(5, 5, 5))
    assert written_data.shape == (1, 250, 250, 10)
    assert np.array_equal(written_data, np.expand_dims(data, 0))

    assert mag.get_tile(1, 1, 6).shape == (1, 32, 32, 1)
    assert np.array_equal(
        mag.get_tile(1, 1, 6)[0, :, :, 0],
        TiffReader("./testoutput/tiled_tiff_dataset/color/1/001_001_006.tif").read(),
    )


def test_open_dataset_without_num_channels_in_properties():
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
        assert data["dataLayers"][0].get("num_channels") is not None


def test_advanced_pattern():
    delete_dir("../testoutput/tiff_dataset_advanced_pattern")
    ds = TiledTiffDataset.create(
        "../testoutput/tiff_dataset_advanced_pattern",
        scale=(1, 1, 1),
        tile_size=(32, 32),
        pattern="{xxxx}/{yyyy}/{zzzz}.tif",
    )
    mag = ds.add_layer("color", Layer.COLOR_TYPE).add_mag("1")
    data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(data)

    assert np.array_equal(mag.read(size=(10, 10, 10)), np.expand_dims(data, 0))


def test_invalid_pattern():

    delete_dir("../testoutput/tiff_invalid_dataset")
    try:
        TiledTiffDataset.create(
            "../testoutput/tiff_invalid_dataset",
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
            "../testoutput/tiff_invalid_dataset",
            scale=(1, 1, 1),
            tile_size=(32, 32),
            pattern="zzzz.tif",
        )
        raise Exception(
            "The test 'test_invalid_pattern' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_properties_with_segmentation():
    input_json_path = "../testdata/complex_property_ds/datasource-properties.json"
    output_json_path = "../testoutput/complex_property_ds/datasource-properties.json"
    properties = WKProperties._from_json(input_json_path)

    # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
    assert properties.data_layers["segmentation"].largest_segment_id == 1000000000
    assert properties.data_layers["segmentation"].mappings == ["larger5um1", "axons", "astrocyte-ge-7", "astrocyte", "mitochondria", "astrocyte-full"]

    # export the json under a new name
    makedirs(dirname(output_json_path), exist_ok=True)
    properties._path = output_json_path
    properties._export_as_json()

    # validate if contents match
    with open(input_json_path) as input_properties:
        input_data = json.load(input_properties)
        for layer in input_data["dataLayers"]:
            # remove the num_channels because they are not part of the original json
            del layer["dataFormat"]

        with open(output_json_path) as output_properties:
            output_data = json.load(output_properties)
            for layer in output_data["dataLayers"]:
                # remove the num_channels because they are not part of the original json
                del layer["num_channels"]

            assert input_data == output_data

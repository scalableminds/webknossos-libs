import filecmp

import numpy as np
from shutil import rmtree, copytree

from wkcuber.api.Dataset import WKDataset, TiffDataset
from os import path, mkdir

from wkcuber.api.Properties import Properties, TiffProperties


def delete_dir(relative_path):
    if path.exists(relative_path) and path.isdir(relative_path):
        rmtree(relative_path)


def test_create_WKDataset_with_layer_and_mag():
    delete_dir("../testoutput/WKDataset")

    ds = WKDataset.create("../testoutput/WKDataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("../testoutput/WKDataset/color/1")
    assert path.exists("../testoutput/WKDataset/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 2


def test_create_WKDataset_with_explicit_header_fields():
    delete_dir("../testoutput/WKDataset_advanced")

    ds = WKDataset.create("../testoutput/WKDataset_advanced", scale=(1, 1, 1))
    ds.add_layer("color", "color", dtype=np.uint16, num_channels=3)

    ds.get_layer("color").add_mag("1", block_len=64, file_len=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("../testoutput/WKDataset_advanced/color/1")
    assert path.exists("../testoutput/WKDataset_advanced/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 2

    assert ds.properties.data_layers["color"].element_class == np.dtype(np.uint16)
    assert (
        ds.properties.data_layers["color"].wkw_resolutions[0].cube_length == 64 * 64
    )  # mag "1"
    assert (
        ds.properties.data_layers["color"].wkw_resolutions[1].cube_length == 32 * 32
    )  # mag "2-2-1" (defaults are used)


def test_create_TiffDataset_with_layer_and_mag():
    # This test would be the same for WKDataset

    delete_dir("../testoutput/TiffDataset")

    ds = WKDataset.create("../testoutput/TiffDataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists("../testoutput/TiffDataset/color/1")
    assert path.exists("../testoutput/TiffDataset/color/2-2-1")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 2


def test_open_wk_dataset():
    ds = WKDataset.open("../testdata/simple_wk_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 1


def test_open_tiff_dataset():
    ds = TiffDataset.open("../testdata/simple_tiff_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 1


def test_slice_read_with_open():
    # This test would be the same for TiffDataset

    wk_slice = WKDataset.open("../testdata/simple_wk_dataset/").get_slice("color", "1")

    assert not wk_slice._is_opened

    with wk_slice.open():
        assert wk_slice._is_opened

        data = wk_slice.read((10, 10, 10))
        assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_slice._is_opened


def test_slice_read_without_open():
    # This test would be the same for TiffDataset

    wk_slice = WKDataset.open("../testdata/simple_wk_dataset/").get_slice("color", "1")

    assert not wk_slice._is_opened

    # 'read()' checks if it was already opened. If not, it opens and closes automatically
    data = wk_slice.read((10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_slice._is_opened


def test_slice_wk_write():
    delete_dir("../testoutput/simple_wk_dataset/")
    copytree("../testdata/simple_wk_dataset/", "../testoutput/simple_wk_dataset/")

    wk_slice = WKDataset.open("../testoutput/simple_wk_dataset/").get_slice(
        "color", "1", size=(100, 100, 100)
    )

    with wk_slice.open():
        np.random.seed(1234)
        write_data = np.random.rand(3, 10, 10, 10).astype(np.uint8)

        wk_slice.write(write_data)

        data = wk_slice.read((10, 10, 10))
        assert np.array_equal(data, write_data)


def test_slice_tiff_write():
    delete_dir("../testoutput/simple_tiff_dataset/")
    copytree("../testdata/simple_tiff_dataset/", "../testoutput/simple_tiff_dataset/")

    tiff_slice = TiffDataset.open("../testoutput/simple_tiff_dataset/").get_slice(
        "color", "1", size=(100, 100, 100)
    )

    with tiff_slice.open():
        np.random.seed(1234)
        write_data = np.random.rand(5, 5, 5).astype(np.uint8)

        tiff_slice.write(np.zeros((5, 5, 5), dtype=np.uint8))

        data = tiff_slice.read((5, 5, 5))
        assert data.shape == (1, 5, 5, 5)  # this dataset has only one channel
        assert np.array_equal(data, np.expand_dims(write_data, 0))


def test_slice_tiff_write_out_of_bounds():
    new_dataset_path = "../testoutput/tiff_slice_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("../testdata/simple_tiff_dataset/", new_dataset_path)

    tiff_slice = TiffDataset.open(new_dataset_path).get_slice(
        "color", "1", size=(100, 100, 100)
    )

    with tiff_slice.open():
        try:
            tiff_slice.write(
                np.zeros((200, 200, 5), dtype=np.uint8)
            )  # this is bigger than the bounding_box
            raise Exception(
                "The test 'test_slice_tiff_write_out_of_bounds' did not throw an exception even though it should"
            )
        except AssertionError:
            pass


def test_tiff_write_out_of_bounds():
    new_dataset_path = "../testoutput/simple_tiff_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("../testdata/simple_tiff_dataset/", new_dataset_path)

    ds = TiffDataset.open(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (265, 265, 10)
    mag_dataset.write(
        np.zeros((300, 300, 15), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (300, 300, 15)


def test_wk_write_out_of_bounds():
    new_dataset_path = "../testoutput/simple_wk_dataset_out_of_bounds/"

    delete_dir(new_dataset_path)
    copytree("../testdata/simple_wk_dataset/", new_dataset_path)

    ds = WKDataset.open(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 24)
    mag_dataset.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 48)


def test_update_new_bounding_box_offset():
    # This test would be the same for WKDataset

    delete_dir("../testoutput/TiffDataset")

    ds = TiffDataset.create("../testoutput/TiffDataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", "color").add_mag("1")

    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (-1, -1, -1)

    write_data = np.random.rand(10, 10, 10).astype(np.uint8)
    mag.write(
        write_data, offset=(10, 10, 10)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (10, 10, 10)

    mag.write(
        write_data, offset=(5, 5, 20)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (5, 5, 10)


def test_tiff_write_multi_channel_uint8():
    dataset_path = "../testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", "color", num_channels=3).add_mag("1")

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
    dataset_path = "../testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", "color", num_channels=3).add_mag("1")

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
    dataset_path = "../testoutput/tiff_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", "color", num_channels=3, dtype=np.uint16).add_mag(
        "1"
    )

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
    dataset_path = "../testoutput/wk_multichannel/"
    delete_dir(dataset_path)

    ds_tiff = WKDataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds_tiff.add_layer("color", "color", num_channels=3, dtype=np.uint16).add_mag(
        "1"
    )

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
    filename = "../testoutput/Empty_WKDataset"
    delete_dir(filename)

    mag = (
        WKDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", "color")
        .add_mag("1")
    )
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_empty_read():
    filename = "../testoutput/Empty_TiffDataset"
    delete_dir(filename)

    mag = (
        TiffDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", "color")
        .add_mag("1")
    )
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_read_padded_data():
    filename = "../testoutput/Empty_TiffDataset"
    delete_dir(filename)

    mag = (
        TiffDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", "color", num_channels=3)
        .add_mag("1")
    )
    # there are no tiffs yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_wk_read_padded_data():
    filename = "../testoutput/Empty_WKDataset"
    delete_dir(filename)

    mag = (
        WKDataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", "color", num_channels=3)
        .add_mag("1")
    )
    # there are no tiffs yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_read_and_write_of_properties():
    destination_path = "../testoutput/read_write_properties/"
    delete_dir(destination_path)
    source_file_name = "../testdata/simple_tiff_dataset/datasource-properties.json"
    destination_file_name = destination_path + "datasource-properties.json"

    imported_properties = TiffProperties.from_json(source_file_name)
    imported_properties.path = destination_file_name
    mkdir(destination_path)
    imported_properties.export_as_json()

    filecmp.cmp(source_file_name, destination_file_name)


def test_num_channel_mismatch_assertion():
    delete_dir("../testoutput/WKDataset")

    ds = WKDataset.create("../testoutput/WKDataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", "color", num_channels=1).add_mag(
        "1"
    )  # num_channel=1 is also the default

    write_data = np.random.rand(3, 10, 10, 10).astype(np.uint8)  # 3 channel

    try:
        mag.write(write_data)  # there is a mismatch between the number of channels
        raise Exception(
            "The test 'test_num_channel_mismatch_assertion' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_layer():
    # This test would be the same for TiffDataset

    delete_dir("../testoutput/WKDataset")

    ds = WKDataset.create("../testoutput/WKDataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer("color", "color", dtype=np.uint8, num_channels=1)
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer("color", "color", dtype=np.uint8, num_channels=1)
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    try:
        # layer did exist before but with another 'dtype' (this would work the same for 'category' and 'num_channels')
        layer = ds.get_or_add_layer("color", "color", dtype=np.uint16, num_channels=1)

        raise Exception(
            "The test 'test_get_or_add_layer' did not throw an exception even though it should"
        )
    except AssertionError:
        pass


def test_get_or_add_mag_for_wk():
    delete_dir("../testoutput/WKDataset")

    layer = WKDataset.create("../testoutput/WKDataset", scale=(1, 1, 1)).add_layer(
        "color", "color"
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
    delete_dir("../testoutput/WKDataset")

    layer = TiffDataset.create("../testoutput/WKDataset", scale=(1, 1, 1)).add_layer(
        "color", "color"
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

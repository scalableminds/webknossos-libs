import numpy as np
from shutil import rmtree, copytree

from wkcuber.api.Dataset import WKDataset, TiffDataset
from os import path

from wkcuber.api.Slice import WKSlice, TiffSlice


def delete_dir(relatve_path):
    dirname = path.join(path.dirname(__file__), relatve_path)
    if path.exists(dirname) and path.isdir(dirname):
        rmtree(dirname)


def test_create_WKDataset_with_layer_and_mag():
    delete_dir("../testoutput/WKDataset")

    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testoutput/WKDataset")
    ds = WKDataset.create(filename, [1])
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists(path.join(dirname, "../testoutput/WKDataset/color/1"))
    assert path.exists(path.join(dirname, "../testoutput/WKDataset/color/2-2-1"))

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 2


def test_create_TiffDataset_with_layer_and_mag():
    delete_dir("../testoutput/TiffDataset")

    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testoutput/TiffDataset")
    ds = WKDataset.create(filename, [1])
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert path.exists(path.join(dirname, "../testoutput/TiffDataset/color/1"))
    assert path.exists(
        path.join(dirname, "../testoutput/TiffDataset/color/2-2-1")
    )

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 2


def test_open_wk_dataset():
    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testdata/simple_wk_dataset")
    ds = WKDataset.open(filename)

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_resolutions) == 1


def test_slice_read_with_open():
    dirname = path.dirname(__file__)
    mag_filename = path.join(dirname, "../testdata/simple_wk_dataset/color/1")
    slice = WKSlice(mag_filename)

    assert not slice._is_opened

    with slice.open():
        assert slice._is_opened

        data = slice.read((10, 10, 10))
        assert data.shape == (3, 10, 10, 10)  # three channel

    assert not slice._is_opened


def test_slice_read_without_open():
    dirname = path.dirname(__file__)
    mag_filename = path.join(dirname, "../testdata/simple_wk_dataset/color/1")
    slice = WKSlice(mag_filename)

    assert not slice._is_opened

    # 'read()' checks if it was already opened. If not, it opens and closes automatically
    data = slice.read((10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel

    assert not slice._is_opened


def test_slice_wk_write():
    dirname = path.dirname(__file__)
    old_dataset_path = path.join(dirname, "../testdata/simple_wk_dataset/")
    new_dataset_path = path.join(dirname, "../testoutput/simple_wk_dataset/")
    mag_filename = path.join(new_dataset_path, "color/1")

    delete_dir(new_dataset_path)

    copytree(old_dataset_path, new_dataset_path)

    slice = WKSlice(mag_filename, size=(100, 100, 100))

    with slice.open():
        data = slice.read((10, 10, 10))
        assert not np.array_equal(data, np.zeros((3, 10, 10, 10)))

        slice.write(np.zeros((3, 10, 10, 10), dtype=np.uint8))

        data = slice.read((10, 10, 10))
        assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_slice_tiff_write():
    dirname = path.dirname(__file__)
    old_dataset_path = path.join(dirname, "../testdata/simple_tiff_dataset/")
    new_dataset_path = path.join(dirname, "../testoutput/simple_tiff_dataset/")
    mag_filename = path.join(new_dataset_path, "color/1")

    delete_dir(new_dataset_path)

    copytree(old_dataset_path, new_dataset_path)

    slice = TiffSlice(mag_filename, size=(100, 100, 100))

    with slice.open():  # no need to pass a header because the default fit for this test
        data = slice.read((5, 5, 5))
        assert not np.array_equal(data, np.zeros((5, 5, 5)))

        slice.write(np.zeros((5, 5, 5), dtype=np.uint8))

        data = slice.read((5, 5, 5))
        assert np.array_equal(data, np.zeros((1, 5, 5, 5)))


def test_slice_tiff_write_out_of_bounds():
    dirname = path.dirname(__file__)
    old_dataset_path = path.join(dirname, "../testdata/simple_tiff_dataset/")
    new_dataset_path = path.join(
        dirname, "../testoutput/tiff_slice_dataset_out_of_bounds/"
    )
    mag_filename = path.join(new_dataset_path, "color/1")

    delete_dir(new_dataset_path)

    copytree(old_dataset_path, new_dataset_path)

    slice = TiffSlice(mag_filename, size=(100, 100, 100))

    with slice.open():
        try:
            slice.write(
                np.zeros((200, 200, 5), dtype=np.uint8)
            )  # this is bigger than the bounding_box
            raise AssertionError(
                "The test 'test_slice_tiff_write_out_of_bounds' did not throw an exception even though it should"
            )
        except Exception:
            pass


def test_tiff_write_out_of_bounds():
    dirname = path.dirname(__file__)
    old_dataset_path = path.join(dirname, "../testdata/simple_tiff_dataset/")
    new_dataset_path = path.join(
        dirname, "../testoutput/simple_tiff_dataset_out_of_bounds/"
    )

    delete_dir(new_dataset_path)

    copytree(old_dataset_path, new_dataset_path)

    ds = TiffDataset.open(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (265, 265, 10)
    mag_dataset.write(
        np.zeros((300, 300, 15), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (300, 300, 15)


def test_wk_write_out_of_bounds():
    dirname = path.dirname(__file__)
    old_dataset_path = path.join(dirname, "../testdata/simple_wk_dataset/")
    new_dataset_path = path.join(
        dirname, "../testoutput/simple_wk_dataset_out_of_bounds/"
    )

    delete_dir(new_dataset_path)

    copytree(old_dataset_path, new_dataset_path)

    ds = WKDataset.open(new_dataset_path)
    mag_dataset = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (
        1024,
        1024,
        1024,
    )
    mag_dataset.write(
        np.zeros((3, 1, 1, 2048), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (
        1024,
        1024,
        2048,
    )


def test_tiff_write_multi_channel_uint8():
    dirname = path.dirname(__file__)
    dataset_path = path.join(dirname, "../testoutput/tiff_multichannel/")

    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, [1])
    ds_tiff.add_layer("color", "color", num_channels=3)
    ds_tiff.get_layer("color").add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint8
    data = np.zeros((3, 250, 250, 10), dtype=np.uint8)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i
                data[1, i, j, h] = j
                data[2, i, j, h] = 100

    ds_tiff.get_layer("color").get_mag("1").write(data)


def test_tiff_write_multi_channel_uint16():
    dirname = path.dirname(__file__)
    dataset_path = path.join(dirname, "../testoutput/tiff_multichannel/")

    delete_dir(dataset_path)

    ds_tiff = TiffDataset.create(dataset_path, [1])
    ds_tiff.add_layer("color", "color", num_channels=3, dtype=np.uint16)
    ds_tiff.get_layer("color").add_mag("1")

    # 10 images (z-layers), each 250x250, dtype=np.uint16
    data = np.zeros((3, 250, 250, 10), dtype=np.uint16)
    for h in range(10):
        for i in range(250):
            for j in range(250):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256

    ds_tiff.get_layer("color").get_mag("1").write(data)


def test_wkw_empty_read():
    delete_dir("../testoutput/Empty_WKDataset")

    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testoutput/Empty_WKDataset")

    mag = WKDataset.create(filename, [1]).add_layer("color", "color").add_mag("1")
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_empty_read():
    delete_dir("../testoutput/Empty_TiffDataset")

    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testoutput/Empty_TiffDataset")

    mag = TiffDataset.create(filename, [1]).add_layer("color", "color").add_mag("1")
    data = mag.read(size=(0, 0, 0), offset=(1, 1, 1))

    assert data.shape == (1, 0, 0, 0)


def test_tiff_read_padded_data():
    delete_dir("../testoutput/Empty_TiffDataset")

    dirname = path.dirname(__file__)
    filename = path.join(dirname, "../testoutput/Empty_TiffDataset")

    mag = (
        TiffDataset.create(filename, [1])
        .add_layer("color", "color", num_channels=3)
        .add_mag("1")
    )
    # there are no tiffs yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))

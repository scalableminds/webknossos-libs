import numpy as np
from wkcuber.utils import get_chunks, get_regular_chunks, BufferedSliceWriter
import wkw
from wkcuber.mag import Mag
import os
from shutil import rmtree

BLOCK_LEN = 32


def delete_dir(relative_path):
    if os.path.exists(relative_path) and os.path.isdir(relative_path):
        rmtree(relative_path)


def test_get_chunks():
    source = list(range(0, 48))
    target = list(get_chunks(source, 8))

    assert len(target) == 6
    assert target[0] == list(range(0, 8))


def test_get_regular_chunks():
    target = list(get_regular_chunks(4, 44, 8))

    assert len(target) == 6
    assert list(target[0]) == list(range(0, 8))
    assert list(target[-1]) == list(range(40, 48))


def test_get_regular_chunks_max_inclusive():
    target = list(get_regular_chunks(4, 44, 1))

    assert len(target) == 41
    assert list(target[0]) == list(range(4, 5))
    # The last chunk should include 44
    assert list(target[-1]) == list(range(44, 45))


def test_buffered_slice_writer():
    test_img = np.arange(24 * 24).reshape(24, 24).astype(np.uint16) + 1
    dtype = test_img.dtype
    bbox = {"topleft": (0, 0, 0), "size": (24, 24, 35)}
    origin = [0, 0, 0]
    dataset_dir = "testoutput/buffered_slice_writer"
    layer_name = "color"
    mag = Mag(1)
    dataset_path = os.path.join(dataset_dir, layer_name, mag.to_layer_name())

    delete_dir(dataset_dir)

    with BufferedSliceWriter(dataset_dir, layer_name, dtype, origin, mag=mag) as writer:
        for i in range(13):
            writer.write_slice(i, test_img)
        with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
            try:
                read_data = data.read(origin, (24, 24, 13))
                if read_data[read_data.nonzero()].size != 0:
                    raise AssertionError(
                        "Nothing should be written on the disk. But found data with shape: {}".format(
                            read_data.shape
                        )
                    )
            except wkw.wkw.WKWException:
                pass

        for i in range(13, 32):
            writer.write_slice(i, test_img)
        with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
            read_data = data.read(origin, (24, 24, 32))
            assert np.squeeze(read_data).shape == (24, 24, 32), (
                "The read data should have the shape: (24, 24, 32) "
                "but has a shape of: {}".format(np.squeeze(read_data).shape)
            )
            assert read_data.size == read_data[read_data.nonzero()].size, (
                "The read data contains zeros while the " "written image has no zeros"
            )

        for i in range(32, 35):
            writer.write_slice(i, test_img)

    with wkw.Dataset.open(dataset_path, wkw.Header(dtype)) as data:
        read_data = data.read(origin, (24, 24, 35))
        read_data = np.squeeze(read_data)
        assert read_data.shape == (24, 24, 35), (
            "The read data should have the shape: (24, 24, 35) "
            "but has a shape of: {}".format(np.squeeze(read_data).shape)
        )
        assert read_data.size == read_data[read_data.nonzero()].size, (
            "The read data contains zeros while the " "written image has no zeros"
        )
        test_img_3d = np.zeros((test_img.shape[0], test_img.shape[1], 35))
        for i in np.arange(35):
            test_img_3d[:, :, i] = test_img
        # transpose because the slice writer takes [y, x] data and transposes it to [x, y] before writing
        test_img_3d = np.transpose(test_img_3d, (1, 0, 2))
        # check if the data are correct
        assert np.array_equal(test_img_3d, read_data), (
            "The data from the disk is not the same "
            "as the data that should be written."
        )

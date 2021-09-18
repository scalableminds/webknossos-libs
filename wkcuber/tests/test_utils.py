from pathlib import Path
from shutil import rmtree
from typing import Union

import numpy as np

from webknossos.dataset import Dataset, LayerCategories
from wkcuber.utils import get_chunks, get_regular_chunks, BufferedSliceWriter, BufferedSliceReader
import wkw
from wkcuber.mag import Mag
import os

BLOCK_LEN = 32

TESTOUTPUT_DIR = Path("testoutput")


def delete_dir(relative_path: Union[str, Path]) -> None:
    if os.path.exists(relative_path) and os.path.isdir(relative_path):
        rmtree(relative_path)


def test_get_chunks() -> None:
    source = list(range(0, 48))
    target = list(get_chunks(source, 8))

    assert len(target) == 6
    assert target[0] == list(range(0, 8))


def test_get_regular_chunks() -> None:
    target = list(get_regular_chunks(4, 44, 8))

    assert len(target) == 6
    assert list(target[0]) == list(range(0, 8))
    assert list(target[-1]) == list(range(40, 48))


def test_get_regular_chunks_max_inclusive() -> None:
    target = list(get_regular_chunks(4, 44, 1))

    assert len(target) == 41
    assert list(target[0]) == list(range(4, 5))
    # The last chunk should include 44
    assert list(target[-1]) == list(range(44, 45))


def test_buffered_slice_writer() -> None:
    test_img = np.arange(24 * 24).reshape(24, 24).astype(np.uint16) + 1
    dtype = test_img.dtype
    origin = (0, 0, 0)
    layer_name = "color"
    mag = Mag(1)
    dataset_dir = TESTOUTPUT_DIR / "buffered_slice_writer"
    dataset_path = str(dataset_dir/layer_name/mag.to_layer_name())

    delete_dir(dataset_dir)
    ds = Dataset.create(dataset_dir, scale=(1, 1, 1))
    mag_view = ds.add_layer("color", LayerCategories.COLOR_TYPE, dtype_per_channel=dtype).add_mag(mag)

    with BufferedSliceWriter(mag_view, origin) as writer:
        writer.send(None)
        for i in range(13):
            writer.send(test_img)
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
            writer.send(test_img)
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
            writer.send(test_img)

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
        # check if the data are correct
        assert np.array_equal(test_img_3d, read_data), (
            "The data from the disk is not the same "
            "as the data that should be written."
        )


def test_buffered_slice_writer_along_different_axis(tmp_path: Path) -> None:
    test_cube = (np.random.random((3, 13, 13, 13))*100).astype(np.uint8)
    cube_size_without_channel = test_cube.shape[1:]
    offset = (5, 10, 20)

    for dim in [0, 1, 2]:
        ds = Dataset.create(tmp_path / f"buffered_slice_writer_{dim}", scale=(1, 1, 1))
        mag_view = ds.add_layer("color", LayerCategories.COLOR_TYPE, num_channels=3).add_mag(1)

        with BufferedSliceWriter(mag_view, offset, buffer_size=5, dimension=dim) as writer:
            writer.send(None)  # to start the generator
            for i in range(cube_size_without_channel[dim]):
                if dim == 0:
                    current_slice = test_cube[:, i, :, :]
                elif dim == 1:
                    current_slice = test_cube[:, :, i, :]
                else:  # dim == 2
                    current_slice = test_cube[:, :, :, i]
                writer.send(current_slice)
        assert np.array_equal(mag_view.read(offset=offset, size=cube_size_without_channel), test_cube)


def test_buffered_slice_reader_along_different_axis(tmp_path: Path) -> None:
    test_cube = (np.random.random((3, 13, 13, 13))*100).astype(np.uint8)
    cube_size_without_channel = test_cube.shape[1:]
    offset = (5, 10, 20)

    for dim in [0, 1, 2]:
        ds = Dataset.create(tmp_path / f"buffered_slice_reader_{dim}", scale=(1, 1, 1))
        mag_view = ds.add_layer("color", LayerCategories.COLOR_TYPE, num_channels=3).add_mag(1)
        mag_view.write(test_cube, offset=offset)

        with BufferedSliceReader(mag_view, offset, cube_size_without_channel, buffer_size=5, dimension=dim) as reader:
            i = 0
            for slice_data in reader:
                if dim == 0:
                    original_slice = test_cube[:, i, :, :]
                elif dim == 1:
                    original_slice = test_cube[:, :, i, :]
                else:  # dim == 2
                    original_slice = test_cube[:, :, :, i]
                i += 1

                assert np.array_equal(slice_data, original_slice)

import warnings
from pathlib import Path

import numpy as np
import wkw

from tests.constants import TESTOUTPUT_DIR
from webknossos.dataset import COLOR_CATEGORY, Dataset
from webknossos.geometry import BoundingBox, Mag, Vec3Int
from webknossos.utils import rmtree

# This module effectively tests BufferedSliceWriter and
# BufferedSliceReader (by calling get_buffered_slice_writer
# and get_buffered_slice_reader).


def test_buffered_slice_writer() -> None:
    test_img = np.arange(24 * 24).reshape(24, 24).astype(np.uint16) + 1
    dtype = test_img.dtype
    origin = Vec3Int.zeros()
    layer_name = "color"
    mag = Mag(1)
    dataset_dir = TESTOUTPUT_DIR / "buffered_slice_writer"
    dataset_path = str(dataset_dir / layer_name / mag.to_layer_name())

    rmtree(dataset_dir)
    ds = Dataset(dataset_dir, voxel_size=(1, 1, 1))
    mag_view = ds.add_layer("color", COLOR_CATEGORY, dtype_per_channel=dtype).add_mag(
        mag
    )

    with mag_view.get_buffered_slice_writer(absolute_offset=origin) as writer:
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
    test_cube = (np.random.random((3, 13, 13, 13)) * 100).astype(np.uint8)
    cube_size_without_channel = test_cube.shape[1:]
    offset = Vec3Int(64, 96, 32)

    for dim in [0, 1, 2]:
        ds = Dataset(tmp_path / f"buffered_slice_writer_{dim}", voxel_size=(1, 1, 1))
        mag_view = ds.add_layer(
            "color", COLOR_CATEGORY, num_channels=test_cube.shape[0]
        ).add_mag(1)

        with mag_view.get_buffered_slice_writer(
            absolute_offset=offset, buffer_size=5, dimension=dim
        ) as writer:
            for i in range(cube_size_without_channel[dim]):
                if dim == 0:
                    current_slice = test_cube[:, i, :, :]
                elif dim == 1:
                    current_slice = test_cube[:, :, i, :]
                else:  # dim == 2
                    current_slice = test_cube[:, :, :, i]
                writer.send(current_slice)
        assert np.array_equal(
            mag_view.read(absolute_offset=offset, size=cube_size_without_channel),
            test_cube,
        )


def test_buffered_slice_reader_along_different_axis(tmp_path: Path) -> None:
    test_cube = (np.random.random((3, 13, 13, 13)) * 100).astype(np.uint8)
    cube_size_without_channel = Vec3Int(test_cube.shape[1:])
    offset = Vec3Int(5, 10, 20)

    for dim in [0, 1, 2]:
        ds = Dataset(tmp_path / f"buffered_slice_reader_{dim}", voxel_size=(1, 1, 1))
        mag_view = ds.add_layer("color", COLOR_CATEGORY, num_channels=3).add_mag(1)
        mag_view.write(test_cube, absolute_offset=offset)

        with mag_view.get_buffered_slice_reader(
            buffer_size=5, dimension=dim
        ) as reader_a, mag_view.get_buffered_slice_reader(
            absolute_bounding_box=BoundingBox(offset, cube_size_without_channel),
            buffer_size=5,
            dimension=dim,
        ) as reader_b:
            i = 0
            for slice_data_a, slice_data_b in zip(reader_a, reader_b):
                if dim == 0:
                    original_slice = test_cube[:, i, :, :]
                elif dim == 1:
                    original_slice = test_cube[:, :, i, :]
                else:  # dim == 2
                    original_slice = test_cube[:, :, :, i]
                i += 1

                assert np.array_equal(slice_data_a, original_slice)
                assert np.array_equal(slice_data_b, original_slice)


def test_basic_buffered_slice_writer(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(8, 8, 8))

    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # This escalates the warning to an error

        # Write some slices
        with mag1.get_buffered_slice_writer() as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape)

    assert np.all(data == written_data)


def test_buffered_slice_writer_unaligned(
    tmp_path: Path,
) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(8, 8, 8))

    # Write some data to z=32. We will check that this
    # data is left untouched by the buffered slice writer.
    ones_at_z32 = np.ones((512, 512, 4), dtype=np.uint8)
    ones_offset = (0, 0, 32)
    mag1.write(ones_at_z32, absolute_offset=ones_offset)

    # Allocate some data (~ 8 MB). Note that this will write
    # from z=1 to z=31 (i.e., 31 slices instead of 32 which
    # is the buffer_size with which we configure the BufferedSliceWriter).
    offset = (1, 1, 1)
    shape = (512, 512, 31)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("default", module="webknossos", message=r"\[WARNING\]")
        with mag1.get_buffered_slice_writer(
            absolute_offset=offset, buffer_size=32
        ) as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data = mag1.read(absolute_offset=offset, size=shape)
    assert np.all(
        data == written_data
    ), "Read data is not equal to the data that was just written."

    data_at_z32 = mag1.read(absolute_offset=ones_offset, size=ones_at_z32.shape)
    assert np.all(
        ones_at_z32 == data_at_z32
    ), "The BufferedSliceWriter seems to have overwritten older data."


def test_buffered_slice_writer_should_warn_about_unaligned_usage(
    tmp_path: Path,
) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(8, 8, 8))

    offset = (1, 1, 1)

    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.filterwarnings("default", module="webknossos", message=r"\[WARNING\]")
        # Write some slices
        with mag1.get_buffered_slice_writer(
            absolute_offset=offset, buffer_size=35
        ) as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

        warning1, warning2 = recorded_warnings
        assert issubclass(warning1.category, UserWarning) and "Using an offset" in str(
            warning1.message
        )
        assert issubclass(
            warning2.category, UserWarning
        ) and "Using a buffer size" in str(warning2.message)

    written_data = mag1.read(absolute_offset=offset, size=shape)

    assert np.all(data == written_data)


def test_basic_buffered_slice_writer_multi_shard(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(4, 4, 4))

    # Allocate some data (~ 3 MB) that covers multiple shards (also in z)
    shape = (160, 150, 140)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # This escalates the warning to an error

        # Write some slices
        with mag1.get_buffered_slice_writer() as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape)

    assert np.all(data == written_data)


def test_basic_buffered_slice_writer_multi_shard_multi_channel(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=3
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(4, 4, 4))

    # Allocate some data (~ 3 MB) that covers multiple shards (also in z)
    shape = (3, 160, 150, 140)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Write some slices
    with mag1.get_buffered_slice_writer() as writer:
        for z in range(0, shape[-1]):
            section = data[:, :, :, z]
            writer.send(section)

    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape[1:])

    assert np.all(data == written_data)


def test_buffered_slice_writer_reset_offset(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=1
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), chunks_per_shard=(8, 8, 8))

    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # This escalates the warning to an error

        # Write some slices
        with mag1.get_buffered_slice_writer() as writer:
            for z in range(0, shape[2] - 8):
                section = data[:, :, z]
                writer.send(section)
            writer.reset_offset(absolute_offset=(0, 0, shape[2] - 8))
            for z in range(shape[2] - 8, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape)

    assert np.all(data == written_data)

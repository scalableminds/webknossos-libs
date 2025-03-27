import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

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
    mag = Mag(1)
    dataset_dir = TESTOUTPUT_DIR / "buffered_slice_writer"

    rmtree(dataset_dir)
    ds = Dataset(dataset_dir, voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel=dtype,
        bounding_box=BoundingBox(origin, (24, 24, 35)),
    )
    mag_view = layer.add_mag(mag, shard_shape=(1024, 1024, 32))

    with mag_view.get_buffered_slice_writer(absolute_offset=origin) as writer:
        for i in range(13):
            writer.send(test_img)

        np.testing.assert_array_equal(
            mag_view.read(absolute_offset=origin, size=(24, 24, 13)),
            0,
            err_msg="Nothing should be written on the disk.",
        )

        for i in range(13, 32):
            writer.send(test_img)

        assert np.all(mag_view.read(absolute_offset=origin, size=(24, 24, 32)) != 0), (
            "The read data contains zeros while the written image has no zeros"
        )

        for i in range(32, 35):
            writer.send(test_img)

    read_data = np.squeeze(mag_view.read(absolute_offset=origin, size=(24, 24, 35)))
    assert np.all(read_data != 0), (
        "The read data contains zeros while the written image has no zeros"
    )
    # check if the data are correct
    test_img_3d = np.zeros((test_img.shape[0], test_img.shape[1], 35))
    for i in np.arange(35):
        test_img_3d[:, :, i] = test_img
    assert np.array_equal(test_img_3d, read_data), (
        "The data from the disk is not the same as the data that should be written."
    )


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_buffered_slice_writer_along_different_axis(
    tmp_path: Path, dim: Literal[0, 1, 2]
) -> None:
    test_cube = (np.random.random((3, 13, 13, 13)) * 100).astype(np.uint8)
    cube_size_without_channel = test_cube.shape[1:]
    offset = Vec3Int(64, 96, 32)

    shard_shape = [1024, 1024, 1024]
    shard_shape[dim] = 32

    ds = Dataset(tmp_path / f"buffered_slice_writer_{dim}", voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=test_cube.shape[0],
        bounding_box=BoundingBox(offset, cube_size_without_channel),
    )
    mag_view = layer.add_mag(1, shard_shape=shard_shape)

    with mag_view.get_buffered_slice_writer(
        absolute_offset=offset, buffer_size=5, dimension=dim, allow_unaligned=True
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
        layer = ds.add_layer(
            "color",
            COLOR_CATEGORY,
            num_channels=3,
            bounding_box=BoundingBox(offset, cube_size_without_channel),
        )
        mag_view = layer.add_mag(1, shard_shape=(1024, 1024, 32))
        mag_view.write(test_cube, absolute_offset=offset)

        with (
            mag_view.get_buffered_slice_reader(
                buffer_size=5, dimension=dim
            ) as reader_a,
            mag_view.get_buffered_slice_reader(
                absolute_bounding_box=BoundingBox(offset, cube_size_without_channel),
                buffer_size=5,
                dimension=dim,
            ) as reader_b,
        ):
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
    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
        bounding_box=BoundingBox((0, 0, 0), shape),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), shard_shape=(256, 256, 256))

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
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
        bounding_box=BoundingBox((0, 0, 0), (513, 513, 36)),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), shard_shape=(256, 256, 256))

    # Write some data to z=32. We will check that this
    # data is left untouched by the buffered slice writer.
    ones_at_z32 = np.ones((512, 512, 4), dtype=np.uint8)
    ones_offset = (0, 0, 32)
    mag1.write(ones_at_z32, absolute_offset=ones_offset, allow_unaligned=True)

    # Allocate some data (~ 8 MB). Note that this will write
    # from z=1 to z=31 (i.e., 31 slices instead of 32 which
    # is the buffer_size with which we configure the BufferedSliceWriter).
    offset = (1, 1, 1)
    shape = (512, 512, 31)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with pytest.warns(UserWarning, match=".*align with the datataset's shard shape.*"):
        with mag1.get_buffered_slice_writer(
            absolute_offset=offset, buffer_size=32, allow_unaligned=True
        ) as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data = mag1.read(absolute_offset=offset, size=shape)
    assert np.all(data == written_data), (
        "Read data is not equal to the data that was just written."
    )

    data_at_z32 = mag1.read(absolute_offset=ones_offset, size=ones_at_z32.shape)
    assert np.all(ones_at_z32 == data_at_z32), (
        "The BufferedSliceWriter seems to have overwritten older data."
    )


def test_buffered_slice_writer_should_raise_unaligned_usage(
    tmp_path: Path,
) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
        bounding_box=BoundingBox((0, 0, 0), (513, 513, 33)),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), shard_shape=(256, 256, 256))

    offset = (1, 1, 1)

    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=".*Using an offset that doesn't align with the datataset's shard shape.*",
    ):
        # Write some slices
        with mag1.get_buffered_slice_writer(
            absolute_offset=offset, buffer_size=35
        ) as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)


def test_basic_buffered_slice_writer_multi_shard(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
        bounding_box=BoundingBox((0, 0, 0), (160, 150, 140)),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), shard_shape=(128, 128, 128))
    assert mag1.info.shard_shape[2] == 32 * 4

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

    written_data = mag1.read(absolute_offset=(0, 0, 0), size=shape).squeeze()

    np.testing.assert_array_equal(data, written_data)


def test_basic_buffered_slice_writer_multi_shard_multi_channel(tmp_path: Path) -> None:
    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=3,
        bounding_box=BoundingBox((0, 0, 0), (160, 150, 140)),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 32), shard_shape=(128, 128, 128))

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
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
        bounding_box=BoundingBox((0, 0, 0), (512, 512, 40)),
    )
    mag1 = layer.add_mag("1", chunk_shape=(32, 32, 8), shard_shape=(256, 256, 8))

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
            writer.reset_offset(absolute_offset=(0, 0, shape[2]))
            for z in range(shape[2] - 8, shape[2]):
                section = data[:, :, z]
                writer.send(section)

    written_data_before_reset = mag1.read(
        absolute_offset=(0, 0, 0), size=(shape[0], shape[1], shape[2] - 8)
    )
    written_data_after_reset = mag1.read(
        absolute_offset=(0, 0, shape[2]), size=(shape[0], shape[1], 8)
    )

    written_data = np.concatenate(
        (written_data_before_reset, written_data_after_reset), axis=3
    )

    assert np.all(data == written_data)


def test_buffered_slice_writer_resize_error(tmp_path: Path) -> None:
    # Allocate some data (~ 8 MB)
    shape = (512, 512, 32)
    data = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Create DS
    dataset = Dataset(tmp_path, voxel_size=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel="uint8",
        num_channels=1,
    )  # intentionally not setting a bounding box
    mag1 = layer.add_mag("1")

    with pytest.raises(ValueError):
        # Write some slices
        with mag1.get_buffered_slice_writer() as writer:
            for z in range(0, shape[2]):
                section = data[:, :, z]
                writer.send(section)

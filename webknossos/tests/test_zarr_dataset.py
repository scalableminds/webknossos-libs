import itertools
import warnings
from pathlib import Path
from shutil import copytree, rmtree
from typing import Generator

import numpy as np
import pytest

from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY, Dataset, MagView
from webknossos.geometry import BoundingBox, Mag, Vec3Int

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


def delete_dir(relative_path: Path) -> None:
    if relative_path.exists() and relative_path.is_dir():
        rmtree(relative_path)


def get_multichanneled_data(dtype: type) -> np.ndarray:
    data = np.zeros((3, 250, 200, 10), dtype=dtype)
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256
    return data


def assure_exported_properties(ds: Dataset) -> None:
    reopened_ds = Dataset.open(ds.path)
    assert (
        ds._properties == reopened_ds._properties
    ), "The properties did not match after reopening the dataset. This might indicate that the properties were not exported after they were changed in memory."


def test_create_dataset_with_layer_and_mag() -> None:
    delete_dir(TESTOUTPUT_DIR / "zarr_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "zarr_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color", data_format="zarr")

    ds.get_layer("color").add_mag("1", chunks_per_shard=Vec3Int.full(1))
    ds.get_layer("color").add_mag("2-2-1", chunks_per_shard=Vec3Int.full(1))

    assert (TESTOUTPUT_DIR / "zarr_dataset" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "zarr_dataset" / "color" / "2-2-1").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assure_exported_properties(ds)


def test_open_dataset() -> None:
    ds = Dataset.open(TESTDATA_DIR / "simple_zarr_dataset")

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1


def test_modify_existing_dataset() -> None:
    delete_dir(TESTOUTPUT_DIR / "simple_zarr_dataset")
    ds1 = Dataset(TESTOUTPUT_DIR / "simple_zarr_dataset", scale=(1, 1, 1))
    ds1.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="float",
        num_channels=1,
        data_format="zarr",
    )

    ds2 = Dataset.open(TESTOUTPUT_DIR / "simple_zarr_dataset")

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        "uint8",
        largest_segment_id=100000,
        data_format="zarr",
    )

    assert (TESTOUTPUT_DIR / "simple_zarr_dataset" / "segmentation").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


def test_view_read() -> None:
    zarr_view = (
        Dataset.open(TESTDATA_DIR / "simple_zarr_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = zarr_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel


def test_view_write() -> None:
    delete_dir(TESTOUTPUT_DIR / "simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / "simple_zarr_dataset", TESTOUTPUT_DIR / "simple_zarr_dataset"
    )

    zarr_view = (
        Dataset.open(TESTOUTPUT_DIR / "simple_zarr_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

    zarr_view.write(write_data)

    data = zarr_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert np.array_equal(data, write_data)


def test_mag_view_write_out_of_bounds() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "simple_zarr_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

    ds = Dataset.open(new_dataset_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


def test_zarr_write_multi_channel_uint16() -> None:
    dataset_path = TESTOUTPUT_DIR / "multichannel"
    delete_dir(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=3,
        dtype_per_layer="uint48",
        data_format="zarr",
    ).add_mag("1", chunks_per_shard=Vec3Int.full(1))

    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read()

    assert np.array_equal(data, written_data)

    assure_exported_properties(ds)


def test_empty_read() -> None:
    filename = TESTOUTPUT_DIR / "empty_zarr_dataset"
    delete_dir(filename)

    mag = (
        Dataset(filename, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format="zarr")
        .add_mag("1", chunks_per_shard=Vec3Int.full(1))
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(absolute_offset=(0, 0, 0), size=(0, 0, 0))


def test_read_padded_data() -> None:
    filename = TESTOUTPUT_DIR / "empty_zarr_dataset"
    delete_dir(filename)

    mag = (
        Dataset(filename, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3, data_format="zarr")
        .add_mag("1", chunks_per_shard=Vec3Int.full(1))
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_get_or_add_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "zarr_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "zarr_dataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="uint8",
        num_channels=1,
        data_format="zarr",
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="uint8",
        num_channels=1,
        data_format="zarr",
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    with pytest.raises(AssertionError):
        # The layer "color" did exist before but with another 'dtype_per_layer' (this would work the same for 'category' and 'num_channels')
        ds.get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_layer="uint16",
            num_channels=1,
            data_format="zarr",
        )

    assure_exported_properties(ds)


def test_get_or_add_layer_idempotence() -> None:
    delete_dir(TESTOUTPUT_DIR / "zarr_dataset")
    ds = Dataset(TESTOUTPUT_DIR / "zarr_dataset", scale=(1, 1, 1))
    ds.get_or_add_layer("color2", "color", np.uint8, data_format="zarr").get_or_add_mag(
        "1", chunks_per_shard=Vec3Int.full(1)
    )
    ds.get_or_add_layer("color2", "color", np.uint8, data_format="zarr").get_or_add_mag(
        "1", chunks_per_shard=Vec3Int.full(1)
    )

    assure_exported_properties(ds)


@pytest.fixture()
def create_dataset(tmp_path: Path) -> Generator[MagView, None, None]:
    ds = Dataset(Path(tmp_path), scale=(2, 2, 1))

    mag = ds.add_layer("color", "color", data_format="zarr").add_mag(
        "2-2-1", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(1)
    )  # cube_size = 8*8 = 64
    yield mag


def test_dataset_conversion() -> None:
    origin_ds_path = TESTOUTPUT_DIR / "conversion" / "origin_zarr"
    converted_ds_path = TESTOUTPUT_DIR / "conversion" / "converted_zarr"

    delete_dir(origin_ds_path)
    delete_dir(converted_ds_path)

    # create example dataset
    origin_ds = Dataset(origin_ds_path, scale=(1, 1, 1))
    seg_layer = origin_ds.add_layer(
        "layer1",
        SEGMENTATION_CATEGORY,
        num_channels=1,
        largest_segment_id=1000000000,
        data_format="zarr",
    )
    seg_layer.add_mag(
        "1", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(1)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8),
    )
    seg_layer.add_mag(
        "2", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(1)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8),
    )
    color_layer = origin_ds.add_layer(
        "layer2", COLOR_CATEGORY, num_channels=3, data_format="zarr"
    )
    color_layer.add_mag(
        "1", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(1)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
    )
    color_layer.add_mag(
        "2", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(1)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8),
    )
    converted_ds = origin_ds.copy_dataset(converted_ds_path)

    assert origin_ds.layers.keys() == converted_ds.layers.keys()
    for layer_name in origin_ds.layers:
        assert (
            origin_ds.layers[layer_name].mags.keys()
            == converted_ds.layers[layer_name].mags.keys()
        )
        for mag in origin_ds.layers[layer_name].mags:
            origin_info = origin_ds.layers[layer_name].mags[mag].info
            converted_info = converted_ds.layers[layer_name].mags[mag].info
            assert origin_info.voxel_type == converted_info.voxel_type
            assert origin_info.num_channels == converted_info.num_channels
            assert origin_info.compression_mode == converted_info.compression_mode
            assert origin_info.chunk_size == converted_info.chunk_size
            assert np.array_equal(
                origin_ds.layers[layer_name].mags[mag].read(),
                converted_ds.layers[layer_name].mags[mag].read(),
            )

    assure_exported_properties(origin_ds)
    assure_exported_properties(converted_ds)


def test_bounding_box_on_disk(
    create_dataset: MagView,  # pylint: disable=redefined-outer-name
) -> None:
    mag = create_dataset

    write_positions = [
        Vec3Int(0, 0, 0),
        Vec3Int(20, 80, 120),
        Vec3Int(1000, 2000, 4000),
    ]
    data_size = Vec3Int(10, 20, 30)
    write_data = (np.random.rand(*data_size) * 255).astype(np.uint8)
    for offset in write_positions:
        mag.write(absolute_offset=offset * mag.mag.to_vec3_int(), data=write_data)

    bounding_boxes_on_disk = list(mag.get_bounding_boxes_on_disk())
    file_size = mag._get_file_dimensions()

    expected_results = set()
    for offset in write_positions:
        range_from = offset // file_size * file_size
        range_to = offset + data_size
        # enumerate all bounding boxes of the current write operation
        x_range = range(
            range_from[0],
            range_to[0],
            file_size[0],
        )
        y_range = range(
            range_from[1],
            range_to[1],
            file_size[1],
        )
        z_range = range(
            range_from[2],
            range_to[2],
            file_size[2],
        )

        for bb_offset in itertools.product(x_range, y_range, z_range):
            expected_results.add(
                BoundingBox(bb_offset, file_size).from_mag_to_mag1(mag.mag)
            )

    assert set(bounding_boxes_on_disk) == expected_results


def test_compression(tmp_path: Path) -> None:
    copytree(Path("testdata", "simple_zarr_dataset"), tmp_path / "dataset")

    mag1 = Dataset.open(tmp_path / "dataset").get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100))

    assert not mag1._is_compressed()
    mag1.compress()
    assert mag1._is_compressed()

    assert np.array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        # writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
        mag1.write(
            (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8),
        )

    assure_exported_properties(mag1.layer.dataset)


def test_rename_layer(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format="zarr")
    mag = layer.add_mag(1, chunks_per_shard=Vec3Int.full(1))
    write_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    mag.write(data=write_data)

    layer.name = "color2"

    assert not (tmp_path / "ds" / "color").exists()
    assert (tmp_path / "ds" / "color2").exists()
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 0
    assert len([l for l in ds._properties.data_layers if l.name == "color2"]) == 1
    assert "color2" in ds.layers.keys()
    assert "color" not in ds.layers.keys()

    # The "mag" object which was created before renaming the layer is still valid
    assert np.array_equal(mag.read()[0], write_data)

    assure_exported_properties(ds)


def test_delete_layer_and_mag(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format="zarr")
    color_layer.add_mag(1, chunks_per_shard=Vec3Int.full(1))
    color_layer.add_mag(2, chunks_per_shard=Vec3Int.full(1))
    ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
        data_format="zarr",
    )
    assert "color" in ds.layers
    assert "segmentation" in ds.layers
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 1
    assert len([l for l in ds._properties.data_layers if l.name == "segmentation"]) == 1
    assert len(color_layer._properties.wkw_resolutions) == 2

    color_layer.delete_mag(1)
    assert len(color_layer._properties.wkw_resolutions) == 1
    assert (
        len(
            [
                m
                for m in color_layer._properties.wkw_resolutions
                if Mag(m.resolution) == Mag(2)
            ]
        )
        == 1
    )

    ds.delete_layer("color")
    assert "color" not in ds.layers
    assert "segmentation" in ds.layers
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 0
    assert len([l for l in ds._properties.data_layers if l.name == "segmentation"]) == 1

    assure_exported_properties(ds)

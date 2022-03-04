import itertools
import json
import os
import pickle
import warnings
from os.path import join
from pathlib import Path
from shutil import copytree, rmtree
from typing import Generator, Tuple, cast

import numpy as np
import pytest

from webknossos.dataset import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    Dataset,
    MagView,
    SegmentationLayer,
    View,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.properties import (
    DatasetProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
    SegmentationLayerProperties,
    dataset_converter,
)
from webknossos.geometry import BoundingBox, Mag, Vec3Int
from webknossos.utils import get_executor_for_args, named_partial, snake_to_camel_case

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


@pytest.fixture(autouse=True, scope="function")
def allow_deprecations() -> Generator:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", module="webknossos", message=r"\[DEPRECATION\]"
        )
        yield


def delete_dir(relative_path: Path) -> None:
    if relative_path.exists() and relative_path.is_dir():
        rmtree(relative_path)


def chunk_job(args: Tuple[View, int]) -> None:
    (view, _i) = args
    # increment the color value of each voxel
    data = view.read(size=view.size)
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    view.write(data)


def advanced_chunk_job(args: Tuple[View, int], dtype: type) -> None:
    view, _i = args

    # write different data for each chunk (depending on the global_offset of the chunk)
    data = view.read(size=view.size)
    data = np.ones(data.shape, dtype=dtype) * dtype(sum(view.global_offset))
    view.write(data)


def for_each_chunking_with_wrong_chunk_size(view: View) -> None:
    with get_executor_for_args(None) as executor:
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_size=(0, 64, 64),
                executor=executor,
            )
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_size=(16, 64, 64),
                executor=executor,
            )
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_size=(100, 64, 64),
                executor=executor,
            )


def for_each_chunking_advanced(ds: Dataset, view: View) -> None:
    with get_executor_for_args(None) as executor:
        func = named_partial(advanced_chunk_job, dtype=np.uint8)
        view.for_each_chunk(
            func,
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
        chunk = ds.get_layer("color").get_mag("1").get_view(size=size, offset=offset)
        chunk_data = chunk.read(size=chunk.size)
        assert np.array_equal(
            np.ones(chunk_data.shape, dtype=np.uint8)
            * np.uint8(sum(chunk.global_offset)),
            chunk_data,
        )


def copy_and_transform_job(args: Tuple[View, View, int], name: str, val: int) -> None:
    (source_view, target_view, _i) = args
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


def assure_exported_properties(ds: Dataset) -> None:
    reopened_ds = Dataset.open(ds.path)
    assert (
        ds._properties == reopened_ds._properties
    ), "The properties did not match after reopening the dataset. This might indicate that the properties were not exported after they were changed in memory."


def test_create_dataset_with_layer_and_mag() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert (TESTOUTPUT_DIR / "wk_dataset" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "wk_dataset" / "color" / "2-2-1").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assure_exported_properties(ds)


def test_create_dataset_with_explicit_header_fields() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset_advanced")

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset_advanced", scale=(1, 1, 1))
    ds.add_layer("color", COLOR_CATEGORY, dtype_per_layer="uint48", num_channels=3)

    ds.get_layer("color").add_mag("1", block_len=64, file_len=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert (TESTOUTPUT_DIR / "wk_dataset_advanced" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "wk_dataset_advanced" / "color" / "2-2-1").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assert ds.get_layer("color").dtype_per_channel == np.dtype("uint16")
    assert ds.get_layer("color")._properties.element_class == "uint48"
    assert ds.get_layer("color").get_mag(1).header.block_len == 64
    assert ds.get_layer("color").get_mag(1).header.file_len == 64
    assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
    assert (
        ds.get_layer("color").get_mag("2-2-1").header.block_len == 32
    )  # defaults are used
    assert (
        ds.get_layer("color").get_mag("2-2-1").header.file_len == 32
    )  # defaults are used
    assert ds.get_layer("color").get_mag("2-2-1")._properties.cube_length == 32 * 32

    assure_exported_properties(ds)


def test_open_dataset() -> None:
    ds = Dataset.open(TESTDATA_DIR / "simple_wk_dataset")

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1


def test_modify_existing_dataset() -> None:
    delete_dir(TESTOUTPUT_DIR / "simple_wk_dataset")
    ds1 = Dataset(TESTOUTPUT_DIR / "simple_wk_dataset", scale=(1, 1, 1))
    ds1.add_layer("color", COLOR_CATEGORY, dtype_per_layer="float", num_channels=1)

    ds2 = Dataset.open(TESTOUTPUT_DIR / "simple_wk_dataset")

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        "uint8",
        largest_segment_id=100000,
    )

    assert (TESTOUTPUT_DIR / "simple_wk_dataset" / "segmentation").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


def test_view_read() -> None:
    wk_view = (
        Dataset.open(TESTDATA_DIR / "simple_wk_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = wk_view.read(size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel


def test_view_write() -> None:
    delete_dir(TESTOUTPUT_DIR / "simple_wk_dataset")
    copytree(TESTDATA_DIR / "simple_wk_dataset", TESTOUTPUT_DIR / "simple_wk_dataset")

    wk_view = (
        Dataset.open(TESTOUTPUT_DIR / "simple_wk_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

    wk_view.write(write_data)

    data = wk_view.read(size=(10, 10, 10))
    assert np.array_equal(data, write_data)


def test_view_write_out_of_bounds() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "wk_view_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_wk_dataset", new_dataset_path)

    view = (
        Dataset.open(new_dataset_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    with pytest.raises(AssertionError):
        view.write(
            np.zeros((200, 200, 5), dtype=np.uint8)
        )  # this is bigger than the bounding_box


def test_mag_view_write_out_of_bounds() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "simple_wk_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_wk_dataset", new_dataset_path)

    ds = Dataset.open(new_dataset_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


def test_mag_view_write_out_of_bounds_mag2() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "simple_wk_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_wk_dataset", new_dataset_path)

    ds = Dataset.open(new_dataset_path)
    mag_view = ds.get_layer("color").get_or_add_mag("2-2-1")

    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8), (10, 10, 10)
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)
    assert tuple(ds.get_layer("color").bounding_box.size) == (120, 24, 58)

    assure_exported_properties(ds)


def test_update_new_bounding_box_offset() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY).add_mag("1")

    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(
        write_data, offset=(10, 10, 10)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert tuple(ds.get_layer("color").bounding_box.topleft) == (10, 10, 10)
    assert tuple(ds.get_layer("color").bounding_box.size) == (10, 10, 10)

    mag.write(
        write_data, offset=(5, 5, 20)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert tuple(ds.get_layer("color").bounding_box.topleft) == (5, 5, 10)
    assert tuple(ds.get_layer("color").bounding_box.size) == (15, 15, 20)

    assure_exported_properties(ds)


def test_write_multi_channel_uint8() -> None:
    dataset_path = TESTOUTPUT_DIR / "multichannel"
    delete_dir(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY, num_channels=3).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    ds.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 200, 10)))

    assure_exported_properties(ds)


def test_wk_write_multi_channel_uint16() -> None:
    dataset_path = TESTOUTPUT_DIR / "multichannel"
    delete_dir(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, dtype_per_layer="uint48"
    ).add_mag("1")

    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read(size=(250, 200, 10))

    assert np.array_equal(data, written_data)

    assure_exported_properties(ds)


def test_empty_read() -> None:
    filename = TESTOUTPUT_DIR / "empty_wk_dataset"
    delete_dir(filename)

    mag = (
        Dataset(filename, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY)
        .add_mag("1")
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(offset=(0, 0, 0), size=(0, 0, 0))


def test_read_padded_data() -> None:
    filename = TESTOUTPUT_DIR / "empty_wk_dataset"
    delete_dir(filename)

    mag = (
        Dataset(filename, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3)
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_num_channel_mismatch_assertion() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY, num_channels=1).add_mag(
        "1"
    )  # num_channel=1 is also the default

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)  # 3 channels

    with pytest.raises(AssertionError):
        mag.write(write_data)  # there is a mismatch between the number of channels

    assure_exported_properties(ds)


def test_get_or_add_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color", COLOR_CATEGORY, dtype_per_layer="uint8", num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color", COLOR_CATEGORY, dtype_per_layer="uint8", num_channels=1
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
        )

    assure_exported_properties(ds)


def test_get_or_add_layer_idempotence() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")
    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    ds.get_or_add_layer("color2", "color", np.uint8).get_or_add_mag("1")
    ds.get_or_add_layer("color2", "color", np.uint8).get_or_add_mag("1")

    assure_exported_properties(ds)


def test_get_or_add_mag() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    layer = Dataset(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1)).add_layer(
        "color", COLOR_CATEGORY
    )

    assert Mag(1) not in layer.mags.keys()

    # The mag did not exist before
    mag = layer.get_or_add_mag("1", block_len=32, file_len=32, compress=False)
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"

    # The mag did exist before
    layer.get_or_add_mag("1", block_len=32, file_len=32, compress=False)
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"

    with pytest.raises(AssertionError):
        # The mag "1" did exist before but with another 'block_len' (this would work the same for 'file_len' and 'block_type')
        layer.get_or_add_mag("1", block_len=64, file_len=32, compress=False)

    assure_exported_properties(layer.dataset)


def test_open_dataset_without_num_channels_in_properties() -> None:
    delete_dir(TESTOUTPUT_DIR / "old_wk_dataset")
    copytree(TESTDATA_DIR / "old_wk_dataset", TESTOUTPUT_DIR / "old_wk_dataset")

    with open(
        TESTOUTPUT_DIR / "old_wk_dataset" / "datasource-properties.json",
        encoding="utf-8",
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") is None

    ds = Dataset.open(TESTOUTPUT_DIR / "old_wk_dataset")
    assert ds.get_layer("color").num_channels == 1
    ds._export_as_json()

    with open(
        TESTOUTPUT_DIR / "old_wk_dataset" / "datasource-properties.json",
        encoding="utf-8",
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("numChannels") == 1

    assure_exported_properties(ds)


def test_largest_segment_id_requirement() -> None:
    path = TESTOUTPUT_DIR / "largest_segment_id"
    delete_dir(path)
    ds = Dataset(path, scale=(10, 10, 10))

    with pytest.raises(AssertionError):
        ds.add_layer("segmentation", SEGMENTATION_CATEGORY)

    largest_segment_id = 10
    ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=largest_segment_id,
    ).add_mag(Mag(1))

    ds = Dataset.open(path)
    assert (
        cast(SegmentationLayer, ds.get_layer("segmentation")).largest_segment_id
        == largest_segment_id
    )

    assure_exported_properties(ds)


def test_properties_with_segmentation() -> None:
    delete_dir(TESTOUTPUT_DIR / "complex_property_ds")
    copytree(
        TESTDATA_DIR / "complex_property_ds",
        TESTOUTPUT_DIR / "complex_property_ds",
    )

    input_path = TESTOUTPUT_DIR / "complex_property_ds"

    with open(input_path / "datasource-properties.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        ds_properties = dataset_converter.structure(data, DatasetProperties)

        # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
        segmentation_layer = cast(
            SegmentationLayerProperties,
            [l for l in ds_properties.data_layers if l.name == "segmentation"][0],
        )
        assert segmentation_layer.largest_segment_id == 1000000000
        assert segmentation_layer.mappings == [
            "larger5um1",
            "axons",
            "astrocyte-ge-7",
            "astrocyte",
            "mitochondria",
            "astrocyte-full",
        ]

    with open(input_path / "datasource-properties.json", "w", encoding="utf-8") as f:
        # Update the properties on disk (without changing the data)
        json.dump(
            dataset_converter.unstructure(ds_properties),
            f,
            indent=4,
        )

    # validate if contents match
    with open(
        TESTDATA_DIR / "complex_property_ds" / "datasource-properties.json",
        encoding="utf-8",
    ) as input_properties:
        input_data = json.load(input_properties)

        with open(
            input_path / "datasource-properties.json", "r", encoding="utf-8"
        ) as output_properties:
            output_data = json.load(output_properties)
            for layer in output_data["dataLayers"]:
                # remove the num_channels because they are not part of the original json
                if "numChannels" in layer:
                    del layer["numChannels"]

            assert input_data == output_data


def test_chunking_wk(tmp_path: Path) -> None:
    ds = Dataset(Path(tmp_path), scale=(2, 2, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag("1", file_len=8, block_len=8)

    original_data = (np.random.rand(50, 100, 150) * 205).astype(np.uint8)
    mag.write(offset=(70, 80, 90), data=original_data)

    # Test with executor
    with get_executor_for_args(None) as executor:
        mag.for_each_chunk(
            chunk_job,
            chunk_size=(64, 64, 64),
            executor=executor,
        )
    assert np.array_equal(original_data + 50, mag.get_view().read()[0])

    # Reset the data
    mag.write(offset=(70, 80, 90), data=original_data)

    # Test without executor
    mag.for_each_chunk(
        chunk_job,
        chunk_size=(64, 64, 64),
    )
    assert np.array_equal(original_data + 50, mag.get_view().read()[0])

    assure_exported_properties(ds)


def test_chunking_wk_advanced() -> None:
    delete_dir(TESTOUTPUT_DIR / "chunking_dataset_wk_advanced")

    ds = Dataset(TESTOUTPUT_DIR / "chunking_dataset_wk_advanced", scale=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1", block_len=8, file_len=8)
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view(size=(150, 150, 54), offset=(10, 10, 10))

    for_each_chunking_advanced(ds, view)

    assure_exported_properties(ds)


def test_chunking_wk_wrong_chunk_size() -> None:
    delete_dir(TESTOUTPUT_DIR / "chunking_dataset_wk_with_wrong_chunk_size")
    ds = Dataset(
        TESTOUTPUT_DIR / "chunking_dataset_wk_with_wrong_chunk_size", scale=(1, 1, 2)
    )
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1", block_len=8, file_len=8)
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view(size=(256, 256, 256))

    for_each_chunking_with_wrong_chunk_size(view)

    assure_exported_properties(ds)


def test_typing_of_get_mag() -> None:
    ds = Dataset.open(TESTDATA_DIR / "simple_wk_dataset")
    layer = ds.get_layer("color")
    assert layer.get_mag("1") == layer.get_mag(1)
    assert layer.get_mag("1") == layer.get_mag((1, 1, 1))
    assert layer.get_mag("1") == layer.get_mag([1, 1, 1])
    assert layer.get_mag("1") == layer.get_mag(np.array([1, 1, 1]))
    assert layer.get_mag("1") == layer.get_mag(Mag(1))

    assure_exported_properties(ds)


def test_dataset_exist_ok() -> None:
    ds_path = TESTOUTPUT_DIR / "wk_dataset_exist_ok"
    delete_dir(ds_path)

    # dataset does not exists yet
    ds1 = Dataset(ds_path, scale=(1, 1, 1), exist_ok=False)
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", COLOR_CATEGORY)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = Dataset(ds_path, scale=(1, 1, 1), exist_ok=True)
    assert "color" in ds2.layers.keys()

    ds2 = Dataset(ds_path, scale=(1, 1, 1), name="wk_dataset_exist_ok", exist_ok=True)
    assert "color" in ds2.layers.keys()

    with pytest.raises(AssertionError):
        # dataset already exists, but with a different scale
        Dataset(ds_path, scale=(2, 2, 2), exist_ok=True)

    with pytest.raises(AssertionError):
        # dataset already exists, but with a different name
        Dataset(ds_path, scale=(1, 1, 1), name="some different name", exist_ok=True)

    assure_exported_properties(ds1)


def test_changing_layer_bounding_box() -> None:
    delete_dir(TESTOUTPUT_DIR / "test_changing_layer_bounding_box")
    copytree(
        TESTDATA_DIR / "simple_wk_dataset",
        TESTOUTPUT_DIR / "test_changing_layer_bounding_box",
    )

    ds = Dataset.open(TESTOUTPUT_DIR / "test_changing_layer_bounding_box")
    layer = ds.get_layer("color")
    mag = layer.get_mag("1")

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (24, 24, 24)
    original_data = mag.read(size=bbox_size)
    assert original_data.shape == (3, 24, 24, 24)

    layer.bounding_box = layer.bounding_box.with_size(
        [12, 12, 10]
    )  # decrease bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (12, 12, 10)
    less_data = mag.read(size=bbox_size)
    assert less_data.shape == (3, 12, 12, 10)
    assert np.array_equal(original_data[:, :12, :12, :10], less_data)

    layer.bounding_box = layer.bounding_box.with_size(
        [36, 48, 60]
    )  # increase the bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (36, 48, 60)
    more_data = mag.read(size=bbox_size)
    assert more_data.shape == (3, 36, 48, 60)
    assert np.array_equal(more_data[:, :24, :24, :24], original_data)

    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)

    # Move the offset from (0, 0, 0) to (10, 10, 0)
    # Note that the bottom right coordinate of the dataset is still at (24, 24, 24)
    layer.bounding_box = BoundingBox((10, 10, 0), (14, 14, 24))

    new_bbox_offset = ds.get_layer("color").bounding_box.topleft
    new_bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(new_bbox_offset) == (10, 10, 0)
    assert tuple(new_bbox_size) == (14, 14, 24)
    assert np.array_equal(original_data, mag.read((0, 0, 0)))

    assert np.array_equal(
        original_data[:, 10:, 10:, :], mag.read(offset=(10, 10, 0), size=(14, 14, 24))
    )

    # resetting the offset to (0, 0, 0)
    # Note that the size did not change. Therefore, the new bottom right is now at (14, 14, 24)
    layer.bounding_box = BoundingBox((0, 0, 0), new_bbox_size)
    new_data = mag.read()
    assert new_data.shape == (3, 14, 14, 24)
    assert np.array_equal(original_data[:, :14, :14, :], new_data)

    assure_exported_properties(ds)


def test_get_view() -> None:
    delete_dir(TESTOUTPUT_DIR / "get_view_tests")

    ds = Dataset(TESTOUTPUT_DIR / "get_view_tests", scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY).add_mag("1")

    # The dataset is new -> no data has been written.
    # Therefore, the size of the bounding box in the properties.json is (0, 0, 0)

    # Creating this view works because the size is set to (0, 0, 0)
    # However, in practice a view with size (0, 0, 0) would not make sense
    # Sizes that contain "0" are not allowed usually, except for an empty layer
    assert mag.get_view().bounding_box.is_empty()

    with pytest.raises(AssertionError):
        # This view exceeds the bounding box
        mag.get_view(size=(16, 16, 16))

    # read-only-views may exceed the bounding box
    read_only_view = mag.get_view(size=(16, 16, 16), read_only=True)
    assert read_only_view.global_offset == tuple((0, 0, 0))
    assert read_only_view.size == tuple((16, 16, 16))

    with pytest.raises(AssertionError):
        # Trying to get a writable sub-view of a read-only-view is not allowed
        read_only_view.get_view(read_only=False)

    np.random.seed(1234)
    write_data = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    # This operation updates the bounding box of the dataset according to the written data
    mag.write(write_data, offset=(10, 20, 30))

    with pytest.raises(AssertionError):
        # The offset and size default to (0, 0, 0).
        # Sizes that contain "0" are not allowed
        mag.get_view(size=(10, 10, 0))

    assert mag.global_offset == (0, 0, 0)  # MagViews always start at (0, 0, 0)
    assert mag.size == (110, 220, 330)

    # Therefore, creating a view with a size of (16, 16, 16) is now allowed
    wk_view = mag.get_view(size=(16, 16, 16))
    assert wk_view.global_offset == (10, 20, 30)
    assert wk_view.size == (16, 16, 16)

    with pytest.raises(AssertionError):
        # Creating this view does not work because the offset (0, 0, 0) would be outside
        # of the bounding box from the properties.json.
        mag.get_view(size=(26, 36, 46), offset=(0, 0, 0))

    # But setting "read_only=True" still works
    mag.get_view(size=(26, 36, 46), offset=(0, 0, 0), read_only=True)

    # Creating this subview works because the subview is completely inside the 'wk_view'.
    # Note that the offset in "get_view" is always relative to the "global_offset"-attribute of the called view.
    sub_view = wk_view.get_view(offset=(8, 8, 8), size=(8, 8, 8))
    assert sub_view.global_offset == tuple((18, 28, 38))
    assert sub_view.size == tuple((8, 8, 8))

    with pytest.raises(AssertionError):
        # Creating this subview does not work because it is not completely inside the 'wk_view'
        wk_view.get_view(offset=(8, 8, 8), size=(10, 10, 10))

    # Again: read-only is allowed
    wk_view.get_view(offset=(8, 8, 8), size=(10, 10, 10), read_only=True)

    with pytest.raises(AssertionError):
        # negative offsets are not allowed
        mag.get_view(offset=(-1, -2, -3))

    with pytest.raises(AssertionError):
        # The fact that this call fails, might be confusing at first glance, but this is intentional.
        # MagViews always start at (0,0,0), regardless of the bounding box in the properties.
        # In this case the offset of "mag" in the properties is (8, 8, 8).
        # If this operation would return a View that starts already at (0, 0, 0). However, this view does
        # not have a reference to the layer. As a result, write operations that do not update the properties
        # would be allowed.
        mag.get_view(mag.global_offset, mag.size)

    assure_exported_properties(ds)


def test_adding_layer_with_invalid_dtype_per_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "invalid_dtype")

    ds = Dataset(TESTOUTPUT_DIR / "invalid_dtype", scale=(1, 1, 1))
    with pytest.raises(TypeError):
        # this would lead to a dtype_per_channel of "uint10", but that is not a valid dtype
        ds.add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_layer="uint30",
            num_channels=3,
        )
    with pytest.raises(TypeError):
        # "int" is interpreted as "int64", but 64 bit cannot be split into 3 channels
        ds.add_layer("color", COLOR_CATEGORY, dtype_per_layer="int", num_channels=3)
    ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_layer="int", num_channels=4
    )  # "int"/"int64" works with 4 channels

    assure_exported_properties(ds)


def test_adding_layer_with_valid_dtype_per_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "valid_dtype")

    ds = Dataset(TESTOUTPUT_DIR / "valid_dtype", scale=(1, 1, 1))
    ds.add_layer("color1", COLOR_CATEGORY, dtype_per_layer="uint24", num_channels=3)
    ds.add_layer("color2", COLOR_CATEGORY, dtype_per_layer=np.uint8, num_channels=1)
    ds.add_layer("color3", COLOR_CATEGORY, dtype_per_channel=np.uint8, num_channels=3)
    ds.add_layer("color4", COLOR_CATEGORY, dtype_per_channel="uint8", num_channels=3)
    ds.add_layer(
        "seg1",
        SEGMENTATION_CATEGORY,
        dtype_per_channel="float",
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg2",
        SEGMENTATION_CATEGORY,
        dtype_per_channel=float,
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg3",
        SEGMENTATION_CATEGORY,
        dtype_per_channel=float,
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg4",
        SEGMENTATION_CATEGORY,
        dtype_per_channel="double",
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg5",
        SEGMENTATION_CATEGORY,
        dtype_per_channel="float",
        num_channels=3,
        largest_segment_id=100000,
    )

    with open(
        TESTOUTPUT_DIR / "valid_dtype" / "datasource-properties.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        # The order of the layers in the properties equals the order of creation
        assert data["dataLayers"][0]["elementClass"] == "uint24"
        assert data["dataLayers"][1]["elementClass"] == "uint8"
        assert data["dataLayers"][2]["elementClass"] == "uint24"
        assert data["dataLayers"][3]["elementClass"] == "uint24"
        assert data["dataLayers"][4]["elementClass"] == "float"
        assert data["dataLayers"][5]["elementClass"] == "float"
        assert data["dataLayers"][6]["elementClass"] == "float"
        assert data["dataLayers"][7]["elementClass"] == "double"
        assert data["dataLayers"][8]["elementClass"] == "float96"

    reopened_ds = Dataset.open(
        TESTOUTPUT_DIR / "valid_dtype"
    )  # reopen the dataset to check if the data is read from the properties correctly
    assert reopened_ds.get_layer("color1").dtype_per_layer == "uint24"
    assert reopened_ds.get_layer("color2").dtype_per_layer == "uint8"
    assert reopened_ds.get_layer("color3").dtype_per_layer == "uint24"
    assert reopened_ds.get_layer("color4").dtype_per_layer == "uint24"
    # Note that 'float' and 'double' are stored as 'float32' and 'float64'
    assert reopened_ds.get_layer("seg1").dtype_per_layer == "float32"
    assert reopened_ds.get_layer("seg2").dtype_per_layer == "float32"
    assert reopened_ds.get_layer("seg3").dtype_per_layer == "float32"
    assert reopened_ds.get_layer("seg4").dtype_per_layer == "float64"
    assert reopened_ds.get_layer("seg5").dtype_per_layer == "float96"

    assure_exported_properties(ds)


def test_writing_subset_of_compressed_data_multi_channel() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    write_data1 = (np.random.rand(3, 100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3)
        .add_mag("1", block_len=8, file_len=8)
    )
    mag_view.write(write_data1)
    mag_view.compress()

    # open compressed dataset
    compressed_mag = (
        Dataset.open(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("1")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        write_data2 = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
        # Writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
        # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
        compressed_mag.get_view(offset=(50, 60, 70)).write(
            offset=(10, 20, 30), data=write_data2
        )

    assert np.array_equal(
        write_data2, compressed_mag.read(offset=(60, 80, 100), size=(10, 10, 10))
    )  # the new data was written
    assert np.array_equal(
        write_data1[:, :60, :80, :100],
        compressed_mag.read(offset=(0, 0, 0), size=(60, 80, 100)),
    )  # the old data is still there


def test_writing_subset_of_compressed_data_single_channel() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY)
        .add_mag("1", block_len=8, file_len=8)
    )
    mag_view.write(write_data1)
    mag_view.compress()

    # open compressed dataset
    compressed_mag = (
        Dataset.open(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("1")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        write_data2 = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
        # Writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
        # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
        compressed_mag.get_view(offset=(50, 60, 70)).write(
            offset=(10, 20, 30), data=write_data2
        )

    assert np.array_equal(
        write_data2, compressed_mag.read(offset=(60, 80, 100), size=(10, 10, 10))[0]
    )  # the new data was written
    assert np.array_equal(
        write_data1[:60, :80, :100],
        compressed_mag.read(offset=(0, 0, 0), size=(60, 80, 100))[0],
    )  # the old data is still there


def test_writing_subset_of_compressed_data() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    mag_view = (
        Dataset(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY)
        .add_mag("2", block_len=8, file_len=8)
    )
    mag_view.write((np.random.rand(120, 140, 160) * 255).astype(np.uint8))
    mag_view.compress()

    # open compressed dataset
    compressed_mag = (
        Dataset.open(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("2")
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        compressed_mag.write(
            offset=(10, 20, 30),
            data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
        )

    with warnings.catch_warnings():
        # Calling 'write' with unaligned data on compressed data only fails if the warnings are treated as errors.
        warnings.filterwarnings("error")  # This escalates the warning to an error
        with pytest.raises(RuntimeWarning):
            compressed_mag.write(
                relative_offset=(20, 40, 60),
                data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
            )

        assert compressed_mag.bounding_box == BoundingBox(
            topleft=(
                0,
                0,
                0,
            ),
            size=(120 * 2, 140 * 2, 160 * 2),
        )
        # Writing unaligned data to the edge of the bounding box of the MagView does not raise an error.
        # This write operation writes unaligned data into the bottom-right corner of the MagView.
        compressed_mag.write(
            absolute_offset=(128, 128, 128),
            data=(np.random.rand(56, 76, 96) * 255).astype(np.uint8),
        )
        # This also works for normal Views but they only use the bounding box at the time of creation as reference.
        compressed_mag.get_view().write(
            absolute_offset=(128, 128, 128),
            data=(np.random.rand(56, 76, 96) * 255).astype(np.uint8),
        )

        # Writing aligned data does not raise a warning. Therefore, this does not fail with these strict settings.
        compressed_mag.write(data=(np.random.rand(64, 64, 64) * 255).astype(np.uint8))


def test_writing_subset_of_chunked_compressed_data() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    mag_view = (
        Dataset(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY)
        .add_mag("1", block_len=8, file_len=8)
    )
    mag_view.write(write_data1)
    mag_view.compress()

    # open compressed dataset
    compressed_view = (
        Dataset.open(TESTOUTPUT_DIR / "compressed_data")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(100, 200, 300))
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.

        # Easy case:
        # The aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
        write_data2 = (np.random.rand(50, 40, 30) * 255).astype(np.uint8)
        compressed_view.write(offset=(10, 20, 30), data=write_data2)

        # Advanced case:
        # The aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
        compressed_view.write(
            offset=(10, 20, 30),
            data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
        )

    np.array_equal(
        write_data2, compressed_view.read(offset=(10, 20, 30), size=(50, 40, 30))
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_view.read(offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


def test_add_symlink_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset_with_symlink")
    delete_dir(TESTOUTPUT_DIR / "simple_wk_dataset_copy")
    copytree(
        TESTDATA_DIR / "simple_wk_dataset", TESTOUTPUT_DIR / "simple_wk_dataset_copy"
    )
    # Add an additional segmentation layer to the original dataset
    Dataset.open(TESTOUTPUT_DIR / "simple_wk_dataset_copy").add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )

    original_mag = (
        Dataset.open(TESTOUTPUT_DIR / "simple_wk_dataset_copy")
        .get_layer("color")
        .get_mag("1")
    )

    ds = Dataset(TESTOUTPUT_DIR / "wk_dataset_with_symlink", scale=(1, 1, 1))
    # symlink color layer
    symlink_layer = ds.add_symlink_layer(
        TESTOUTPUT_DIR / "simple_wk_dataset_copy" / "color"
    )
    # symlink segmentation layer
    symlink_segmentation_layer = ds.add_symlink_layer(
        TESTOUTPUT_DIR / "simple_wk_dataset_copy" / "segmentation"
    )
    mag = symlink_layer.get_mag("1")

    assert (TESTOUTPUT_DIR / "wk_dataset_with_symlink" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "wk_dataset_with_symlink" / "segmentation").exists()

    assert len(ds.layers) == 2
    assert len(ds.get_layer("color").mags) == 1

    assert cast(SegmentationLayer, symlink_segmentation_layer).largest_segment_id == 999

    # write data in symlink layer
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data)

    assert np.array_equal(mag.read(size=(10, 10, 10)), write_data)
    assert np.array_equal(original_mag.read(size=(10, 10, 10)), write_data)

    assure_exported_properties(ds)


def test_add_symlink_mag(tmp_path: Path) -> None:
    original_ds = Dataset(tmp_path / "original", scale=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8"
    )
    original_layer.add_mag(1).write(
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )
    original_mag_2 = original_layer.add_mag(2)
    original_mag_2.write(data=(np.random.rand(5, 10, 15) * 255).astype(np.uint8))
    original_mag_4 = original_layer.add_mag(4)
    original_mag_4.write(data=(np.random.rand(2, 5, 7) * 255).astype(np.uint8))

    ds = Dataset(tmp_path / "link", scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, dtype_per_channel="uint8")
    layer.add_mag(1).write(
        offset=(6, 6, 6), data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    symlink_mag_2 = layer.add_symlink_mag(original_mag_2)
    _symlink_mag_4 = layer.add_symlink_mag(original_mag_4.path)

    assert (tmp_path / "link" / "color" / "1").exists()
    assert len(layer._properties.wkw_resolutions) == 3

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    # Write data in symlink layer
    # Note: The written data is fully inside the bounding box of the original data.
    # This is important because the bounding box of the foreign layer would not be updated if we use the linked dataset to write outside of its original bounds.
    write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    symlink_mag_2.write(offset=(0, 0, 0), data=write_data)

    assert np.array_equal(symlink_mag_2.read(size=(5, 5, 5))[0], write_data)
    assert np.array_equal(original_layer.get_mag(2).read(size=(5, 5, 5))[0], write_data)

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


def test_add_copy_mag(tmp_path: Path) -> None:
    original_ds = Dataset(tmp_path / "original", scale=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8"
    )
    original_layer.add_mag(1).write(
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )
    original_data = (np.random.rand(5, 10, 15) * 255).astype(np.uint8)
    original_mag_2 = original_layer.add_mag(2)
    original_mag_2.write(data=original_data)

    ds = Dataset(tmp_path / "link", scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, dtype_per_channel="uint8")
    layer.add_mag(1).write(
        offset=(6, 6, 6), data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    copy_mag = layer.add_copy_mag(original_mag_2)

    assert (tmp_path / "link" / "color" / "1").exists()
    assert len(layer._properties.wkw_resolutions) == 2

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    # Write data in copied layer
    write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    copy_mag.write(offset=(0, 0, 0), data=write_data)

    assert np.array_equal(copy_mag.read(size=(5, 5, 5))[0], write_data)
    assert np.array_equal(original_layer.get_mag(2).read()[0], original_data)

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


def test_search_dataset_also_for_long_layer_name() -> None:
    delete_dir(TESTOUTPUT_DIR / "long_layer_name")

    ds = Dataset(TESTOUTPUT_DIR / "long_layer_name", scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY).add_mag("2")

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

    # make sure that reading data still works
    mag.read(relative_offset=(10, 10, 10), size=(10, 10, 10))

    # when opening the dataset, it searches both for the long and the short path
    layer = Dataset.open(TESTOUTPUT_DIR / "long_layer_name").get_layer("color")
    mag = layer.get_mag("2")
    assert np.array_equal(
        mag.read(offset=(10, 10, 10), size=(10, 10, 10)), np.expand_dims(write_data, 0)
    )
    layer.delete_mag("2")

    # Note: 'ds' is outdated (it still contains Mag(2)) because it was opened again and changed.
    assure_exported_properties(layer.dataset)


def test_outdated_dtype_parameter() -> None:
    delete_dir(TESTOUTPUT_DIR / "outdated_dtype")

    ds = Dataset(TESTOUTPUT_DIR / "outdated_dtype", scale=(1, 1, 1))
    with pytest.raises(ValueError):
        ds.get_or_add_layer("color", COLOR_CATEGORY, dtype=np.uint8, num_channels=1)

    with pytest.raises(ValueError):
        ds.add_layer("color", COLOR_CATEGORY, dtype=np.uint8, num_channels=1)


@pytest.mark.parametrize("make_relative", [True, False])
def test_dataset_shallow_copy(make_relative: bool) -> None:
    delete_dir(TESTOUTPUT_DIR / "original_dataset")
    delete_dir(TESTOUTPUT_DIR / "copy_dataset")
    ds = Dataset(TESTOUTPUT_DIR / "original_dataset", (1, 1, 1))
    original_layer_1 = ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_layer=np.uint8, num_channels=1
    )
    original_layer_1.add_mag(1)
    original_layer_1.add_mag("2-2-1")
    original_layer_2 = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype_per_layer=np.uint32,
        largest_segment_id=0,
    )
    original_layer_2.add_mag(4)
    mappings_path = original_layer_2.path / "mappings"
    os.makedirs(mappings_path)
    open(mappings_path / "agglomerate_view.hdf5", "w", encoding="utf-8").close()

    shallow_copy_of_ds = ds.shallow_copy_dataset(
        TESTOUTPUT_DIR / "copy_dataset", make_relative=make_relative
    )
    shallow_copy_of_ds.get_layer("color").add_mag(Mag("4-4-1"))
    assert (
        len(Dataset.open(TESTOUTPUT_DIR / "original_dataset").get_layer("color").mags)
        == 2
    ), "Adding a new mag should not affect the original dataset"
    assert (
        len(Dataset.open(TESTOUTPUT_DIR / "copy_dataset").get_layer("color").mags) == 3
    ), "Expecting all mags from original dataset and new downsampled mag"
    assert os.path.exists(
        TESTOUTPUT_DIR
        / "copy_dataset"
        / "segmentation"
        / "mappings"
        / "agglomerate_view.hdf5"
    ), "Expecting mappings to exist in shallow copy"


def test_dataset_conversion() -> None:
    origin_ds_path = TESTOUTPUT_DIR / "conversion" / "origin_wk"
    converted_ds_path = TESTOUTPUT_DIR / "conversion" / "converted_wk"

    delete_dir(origin_ds_path)
    delete_dir(converted_ds_path)

    # create example dataset
    origin_ds = Dataset(origin_ds_path, scale=(1, 1, 1))
    seg_layer = origin_ds.add_layer(
        "layer1",
        SEGMENTATION_CATEGORY,
        num_channels=1,
        largest_segment_id=1000000000,
    )
    seg_layer.add_mag("1", block_len=8, file_len=16).write(
        offset=(10, 20, 30), data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8)
    )
    seg_layer.add_mag("2", block_len=8, file_len=16).write(
        offset=(5, 10, 15), data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8)
    )
    wk_color_layer = origin_ds.add_layer("layer2", COLOR_CATEGORY, num_channels=3)
    wk_color_layer.add_mag("1", block_len=8, file_len=16).write(
        offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
    )
    wk_color_layer.add_mag("2", block_len=8, file_len=16).write(
        offset=(5, 10, 15), data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8)
    )
    converted_ds = origin_ds.copy_dataset(converted_ds_path)

    assert origin_ds.layers.keys() == converted_ds.layers.keys()
    for layer_name in origin_ds.layers:
        assert (
            origin_ds.layers[layer_name].mags.keys()
            == converted_ds.layers[layer_name].mags.keys()
        )
        for mag in origin_ds.layers[layer_name].mags:
            origin_header = origin_ds.layers[layer_name].mags[mag].header
            converted_header = converted_ds.layers[layer_name].mags[mag].header
            assert origin_header.voxel_type == converted_header.voxel_type
            assert origin_header.num_channels == converted_header.num_channels
            assert origin_header.block_type == converted_header.block_type
            assert origin_header.block_len == converted_header.block_len
            assert np.array_equal(
                origin_ds.layers[layer_name].mags[mag].read(),
                converted_ds.layers[layer_name].mags[mag].read(),
            )

    assure_exported_properties(origin_ds)
    assure_exported_properties(converted_ds)


def test_for_zipped_chunks() -> None:
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_source")
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_target")

    ds = Dataset(TESTOUTPUT_DIR / "zipped_chunking_source", scale=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1")
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    source_view = mag.get_view(size=(256, 256, 256))

    target_mag = (
        Dataset(TESTOUTPUT_DIR / "zipped_chunking_target", scale=(1, 1, 2))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint8",
            num_channels=3,
        )
        .get_or_add_mag("1", block_len=8, file_len=4)
    )

    target_mag.layer.bounding_box = BoundingBox((0, 0, 0), (256, 256, 256))
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

    assure_exported_properties(ds)


def _func_invalid_target_chunk_size_wk(args: Tuple[View, View, int]) -> None:
    (_s, _t, _i) = args


def test_for_zipped_chunks_invalid_target_chunk_size_wk() -> None:
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_source_invalid")

    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    ds = Dataset(TESTOUTPUT_DIR / "zipped_chunking_source_invalid", scale=(1, 1, 1))
    layer1 = ds.get_or_add_layer("color1", COLOR_CATEGORY)
    source_mag_view = layer1.get_or_add_mag(1, block_len=8, file_len=8)

    layer2 = ds.get_or_add_layer("color2", COLOR_CATEGORY)
    target_mag_view = layer2.get_or_add_mag(1, block_len=8, file_len=8)

    source_view = source_mag_view.get_view(size=(300, 300, 300), read_only=True)
    layer2.bounding_box = BoundingBox((0, 0, 0), (300, 300, 300))
    target_view = target_mag_view.get_view()

    with get_executor_for_args(None) as executor:
        for test_case in test_cases_wk:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    func_per_chunk=_func_invalid_target_chunk_size_wk,
                    target_view=target_view,
                    source_chunk_size=test_case,
                    target_chunk_size=test_case,
                    executor=executor,
                )

    assure_exported_properties(ds)


def test_read_only_view() -> None:
    delete_dir(TESTOUTPUT_DIR / "read_only_view")
    ds = Dataset(TESTOUTPUT_DIR / "read_only_view", scale=(1, 1, 1))
    mag = ds.get_or_add_layer("color", COLOR_CATEGORY).get_or_add_mag("1")
    mag.write(
        data=(np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8), offset=(10, 20, 30)
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = (np.random.rand(1, 5, 6, 7) * 255).astype(np.uint8)
    with pytest.raises(AssertionError):
        v_read.write(data=new_data)

    v_write.write(data=new_data)

    assure_exported_properties(ds)


@pytest.fixture()
def create_dataset(tmp_path: Path) -> Generator[MagView, None, None]:
    ds = Dataset(Path(tmp_path), scale=(2, 2, 1))

    mag = ds.add_layer("color", "color").add_mag(
        "2-2-1", block_len=8, file_len=8
    )  # cube_size = 8*8 = 64
    yield mag


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
        mag.write(offset=offset, data=write_data)

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
    copytree(Path("testdata", "simple_wk_dataset"), tmp_path / "dataset")

    mag1 = Dataset.open(tmp_path / "dataset").get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, offset=(60, 80, 100))

    assert not mag1._is_compressed()
    mag1.compress()
    assert mag1._is_compressed()

    assert np.array_equal(
        write_data, mag1.read(offset=(60, 80, 100), size=(10, 20, 30))
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


def test_dataset_view_configuration(tmp_path: Path) -> None:
    ds1 = Dataset(tmp_path, scale=(2, 2, 1))
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is None

    ds1.default_view_configuration = DatasetViewConfiguration(four_bit=True)
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation == None
    assert default_view_configuration.render_missing_data_black == None
    assert default_view_configuration.loading_strategy == None
    assert default_view_configuration.segmentation_pattern_opacity == None
    assert default_view_configuration.zoom == None
    assert default_view_configuration.position == None
    assert default_view_configuration.rotation == None

    # Test if only the set parameters are stored in the properties
    with open(ds1.path / PROPERTIES_FILE_NAME, encoding="utf-8") as f:
        properties = json.load(f)
        assert properties["defaultViewConfiguration"] == {"fourBit": True}

    ds1.default_view_configuration = DatasetViewConfiguration(
        four_bit=True,
        interpolation=False,
        render_missing_data_black=True,
        loading_strategy="PROGRESSIVE_QUALITY",
        segmentation_pattern_opacity=40,
        zoom=0.1,
        position=(12, 12, 12),
        rotation=(1, 2, 3),
    )

    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation == False
    assert default_view_configuration.render_missing_data_black == True
    assert default_view_configuration.loading_strategy == "PROGRESSIVE_QUALITY"
    assert default_view_configuration.segmentation_pattern_opacity == 40
    assert default_view_configuration.zoom == 0.1
    assert default_view_configuration.position == (12, 12, 12)
    assert default_view_configuration.rotation == (1, 2, 3)

    # Test if the data is persisted to disk
    ds2 = Dataset.open(tmp_path)
    default_view_configuration = ds2.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation == False
    assert default_view_configuration.render_missing_data_black == True
    assert default_view_configuration.loading_strategy == "PROGRESSIVE_QUALITY"
    assert default_view_configuration.segmentation_pattern_opacity == 40
    assert default_view_configuration.zoom == 0.1
    assert default_view_configuration.position == (12, 12, 12)
    assert default_view_configuration.rotation == (1, 2, 3)

    # Test camel case
    with open(ds1.path / PROPERTIES_FILE_NAME, encoding="utf-8") as f:
        properties = json.load(f)
        view_configuration_dict = properties["defaultViewConfiguration"]
        for k in view_configuration_dict.keys():
            assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_layer_view_configuration(tmp_path: Path) -> None:
    ds1 = Dataset(tmp_path, scale=(2, 2, 1))
    layer1 = ds1.add_layer("color", COLOR_CATEGORY)
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is None

    layer1.default_view_configuration = LayerViewConfiguration(color=(255, 0, 0))
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha is None
    assert default_view_configuration.intensity_range is None
    assert default_view_configuration.is_inverted is None
    # Test if only the set parameters are stored in the properties
    with open(ds1.path / PROPERTIES_FILE_NAME, encoding="utf-8") as f:
        properties = json.load(f)
        assert properties["dataLayers"][0]["defaultViewConfiguration"] == {
            "color": [255, 0, 0]
        }

    layer1.default_view_configuration = LayerViewConfiguration(
        color=(255, 0, 0),
        alpha=1.0,
        min=55.0,
        intensity_range=(-12.3e1, 123),
        is_inverted=True,
    )
    default_view_configuration = layer1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha == 1.0
    assert default_view_configuration.intensity_range == (-12.3e1, 123)
    assert default_view_configuration.is_inverted == True
    assert default_view_configuration.min == 55.0

    # Test if the data is persisted to disk
    ds2 = Dataset.open(tmp_path)
    default_view_configuration = ds2.get_layer("color").default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha == 1.0
    assert default_view_configuration.intensity_range == (-12.3e1, 123)
    assert default_view_configuration.is_inverted == True
    assert default_view_configuration.min == 55.0

    # Test camel case
    with open(ds2.path / PROPERTIES_FILE_NAME, encoding="utf-8") as f:
        properties = json.load(f)
        view_configuration_dict = properties["dataLayers"][0][
            "defaultViewConfiguration"
        ]
        for k in view_configuration_dict.keys():
            assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_get_largest_segment_id(tmp_path: Path) -> None:
    ds = Dataset(tmp_path, scale=(1, 1, 1))

    segmentation_layer = cast(
        SegmentationLayer,
        ds.add_layer("segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999),
    )
    assert segmentation_layer.largest_segment_id == 999
    segmentation_layer.largest_segment_id = 123
    assert segmentation_layer.largest_segment_id == 123

    assure_exported_properties(ds)


def test_get_or_add_layer_by_type(tmp_path: Path) -> None:
    ds = Dataset(tmp_path, scale=(1, 1, 1))
    assert len(ds.get_segmentation_layers()) == 0
    _ = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )  # adds layer
    assert len(ds.get_segmentation_layers()) == 1
    _ = ds.add_layer(
        "different_segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
    )  # adds another layer
    assert len(ds.get_segmentation_layers()) == 2

    assert len(ds.get_color_layers()) == 0
    _ = ds.add_layer("color", COLOR_CATEGORY)  # adds layer
    assert len(ds.get_color_layers()) == 1
    _ = ds.add_layer("different_color", COLOR_CATEGORY)  # adds another layer
    assert len(ds.get_color_layers()) == 2

    assure_exported_properties(ds)


def test_dataset_name(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "some_name", scale=(1, 1, 1))
    assert ds.name == "some_name"
    ds.name = "other_name"
    assert ds.name == "other_name"

    ds2 = Dataset(
        tmp_path / "some_new_name", scale=(1, 1, 1), name="very important dataset"
    )
    assert ds2.name == "very important dataset"

    assure_exported_properties(ds)


def test_read_bbox(tmp_path: Path) -> None:
    ds = Dataset(tmp_path, scale=(2, 2, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag(1)
    mag.write(
        offset=(10, 20, 30), data=(np.random.rand(50, 60, 70) * 255).astype(np.uint8)
    )

    assert np.array_equal(mag.read(), mag.read_bbox())
    assert np.array_equal(
        mag.read(offset=(20, 30, 40), size=(40, 50, 60)),
        mag.read_bbox(BoundingBox(topleft=(20, 30, 40), size=(40, 50, 60))),
    )


def test_add_copy_layer(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(2, 2, 1))

    # Create dataset to copy data from
    other_ds = Dataset(tmp_path / "other_ds", scale=(2, 2, 1))
    original_color_layer = other_ds.add_layer("color", COLOR_CATEGORY)
    original_color_layer.add_mag(1).write(
        offset=(10, 20, 30), data=(np.random.rand(32, 64, 128) * 255).astype(np.uint8)
    )
    other_ds.add_layer("segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999)

    # Copies the "color" layer from a different dataset
    ds.add_copy_layer(tmp_path / "other_ds" / "color")
    ds.add_copy_layer(tmp_path / "other_ds" / "segmentation")
    assert len(ds.layers) == 2
    assert (
        cast(SegmentationLayer, ds.get_layer("segmentation")).largest_segment_id == 999
    )
    color_layer = ds.get_layer("color")
    assert color_layer.bounding_box == BoundingBox(
        topleft=(10, 20, 30), size=(32, 64, 128)
    )
    assert color_layer.mags.keys() == original_color_layer.mags.keys()
    assert len(color_layer.mags.keys()) >= 1
    for mag in color_layer.mags.keys():
        assert np.array_equal(
            color_layer.get_mag(mag).read(), original_color_layer.get_mag(mag).read()
        )
        # Test if the copied layer contains actual data
        assert np.max(color_layer.get_mag(mag).read()) > 0

    with pytest.raises(IndexError):
        # The dataset already has a layer called "color".
        ds.add_copy_layer(tmp_path / "other_ds" / "color")

    # Test if the changes of the properties are persisted on disk by opening it again
    assert "color" in Dataset.open(tmp_path / "ds").layers.keys()

    assure_exported_properties(ds)


def test_rename_layer(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag(1)
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
    color_layer = ds.add_layer("color", COLOR_CATEGORY)
    color_layer.add_mag(1)
    color_layer.add_mag(2)
    ds.add_layer("segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999)
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


def test_add_layer_like(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(1, 1, 1))
    color_layer1 = ds.add_layer(
        "color1", COLOR_CATEGORY, dtype_per_layer="uint24", num_channels=3
    )
    color_layer1.add_mag(1)
    segmentation_layer1 = cast(
        SegmentationLayer,
        ds.add_layer(
            "segmentation1",
            SEGMENTATION_CATEGORY,
            dtype_per_channel="uint8",
            largest_segment_id=999,
        ),
    )
    segmentation_layer1.add_mag(1)
    color_layer2 = ds.add_layer_like(color_layer1, "color2")
    segmentation_layer2 = cast(
        SegmentationLayer, ds.add_layer_like(segmentation_layer1, "segmentation2")
    )

    assert color_layer1.name == "color1"
    assert color_layer2.name == "color2"
    assert len(color_layer1.mags) == 1
    assert len(color_layer2.mags) == 0
    assert color_layer1.category == color_layer2.category == COLOR_CATEGORY
    assert (
        color_layer1.dtype_per_channel
        == color_layer2.dtype_per_channel
        == np.dtype("uint8")
    )
    assert color_layer1.num_channels == color_layer2.num_channels == 3

    assert segmentation_layer1.name == "segmentation1"
    assert segmentation_layer2.name == "segmentation2"
    assert len(segmentation_layer1.mags) == 1
    assert len(segmentation_layer2.mags) == 0
    assert (
        segmentation_layer1.category
        == segmentation_layer2.category
        == SEGMENTATION_CATEGORY
    )
    assert (
        segmentation_layer1.dtype_per_channel
        == segmentation_layer2.dtype_per_channel
        == np.dtype("uint8")
    )
    assert segmentation_layer1.num_channels == segmentation_layer2.num_channels == 1
    assert (
        segmentation_layer1.largest_segment_id
        == segmentation_layer2.largest_segment_id
        == 999
    )

    assure_exported_properties(ds)


def test_pickle_view(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "ds", scale=(1, 1, 1))
    mag1 = ds.add_layer("color", COLOR_CATEGORY).add_mag(1)

    assert mag1._cached_wkw_dataset is None
    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag1.write(data_to_write)
    assert mag1._cached_wkw_dataset is not None

    pickle.dump(mag1, open(str(tmp_path / "save.p"), "wb"))
    pickled_mag1 = pickle.load(open(str(tmp_path / "save.p"), "rb"))

    # Make sure that the pickled mag can still read data
    assert pickled_mag1._cached_wkw_dataset is None
    assert np.array_equal(
        data_to_write,
        pickled_mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )
    assert pickled_mag1._cached_wkw_dataset is not None

    # Make sure that the attributes of the MagView (not View) still exist
    assert pickled_mag1.layer is not None

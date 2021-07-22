import filecmp
import itertools
import json
import os
from os.path import dirname, join
from pathlib import Path
from typing import Any, Tuple, cast, Generator

import pytest

import numpy as np
from shutil import rmtree, copytree

from wkw import wkw
from wkw.wkw import WKWException

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.api.dataset import Dataset
from os import makedirs

from wkcuber.api.layer import Layer, LayerCategories, SegmentationLayer
from wkcuber.api.mag_view import MagView
from wkcuber.api.properties.dataset_properties import Properties
from wkcuber.api.properties.layer_properties import SegmentationLayerProperties
from wkcuber.api.properties.resolution_properties import Resolution
from wkcuber.api.view import View
from wkcuber.compress import compress_mag_inplace
from wkcuber.mag import Mag
from wkcuber.utils import get_executor_for_args, named_partial

TESTDATA_DIR = Path("testdata")
TESTOUTPUT_DIR = Path("testoutput")


def delete_dir(relative_path: Path) -> None:
    if relative_path.exists() and relative_path.is_dir():
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
        chunk = ds.get_layer("color").get_mag("1").get_view(size=size, offset=offset)
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


def test_create_dataset_with_layer_and_mag() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color")

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert (TESTOUTPUT_DIR / "wk_dataset" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "wk_dataset" / "color" / "2-2-1").exists()

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2


def test_create_dataset_with_explicit_header_fields() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset_advanced")

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset_advanced", scale=(1, 1, 1))
    ds.add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_layer="uint48", num_channels=3
    )

    ds.get_layer("color").add_mag("1", block_len=64, file_len=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert (TESTOUTPUT_DIR / "wk_dataset_advanced" / "color" / "1").exists()
    assert (TESTOUTPUT_DIR / "wk_dataset_advanced" / "color" / "2-2-1").exists()

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 2

    assert ds.properties.data_layers["color"].element_class == "uint48"
    assert (
        cast(
            Resolution, ds.properties.data_layers["color"].wkw_magnifications[0]
        ).cube_length
        == 64 * 64
    )  # mag "1"
    assert ds.properties.data_layers["color"].wkw_magnifications[0].mag == Mag("1")
    assert (
        cast(
            Resolution, ds.properties.data_layers["color"].wkw_magnifications[1]
        ).cube_length
        == 32 * 32
    )  # mag "2-2-1" (defaults are used)
    assert ds.properties.data_layers["color"].wkw_magnifications[1].mag == Mag("2-2-1")


def test_open_dataset() -> None:
    ds = Dataset(TESTDATA_DIR / "simple_wk_dataset")

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1


def test_view_read_with_open() -> None:
    wk_view = (
        Dataset(TESTDATA_DIR / "simple_wk_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    assert not wk_view._is_opened

    with wk_view.open():
        assert wk_view._is_opened

        data = wk_view.read(size=(10, 10, 10))
        assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_view_read_without_open() -> None:
    wk_view = (
        Dataset(TESTDATA_DIR / "simple_wk_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    assert not wk_view._is_opened

    # 'read()' checks if it was already opened. If not, it opens and closes automatically
    data = wk_view.read(size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel

    assert not wk_view._is_opened


def test_view_write() -> None:
    delete_dir(TESTOUTPUT_DIR / "simple_wk_dataset")
    copytree(TESTDATA_DIR / "simple_wk_dataset", TESTOUTPUT_DIR / "simple_wk_dataset")

    wk_view = (
        Dataset(TESTOUTPUT_DIR / "simple_wk_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    with wk_view.open():
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
        Dataset(new_dataset_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(16, 16, 16))
    )

    with view.open():
        with pytest.raises(AssertionError):
            view.write(
                np.zeros((200, 200, 5), dtype=np.uint8)
            )  # this is bigger than the bounding_box


def test_mag_view_write_out_of_bounds() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "simple_wk_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_wk_dataset", new_dataset_path)

    ds = Dataset(new_dataset_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 48)


def test_mag_view_write_out_of_bounds_mag2() -> None:
    new_dataset_path = TESTOUTPUT_DIR / "simple_wk_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_wk_dataset", new_dataset_path)

    ds = Dataset(new_dataset_path)
    mag_view = ds.get_layer("color").get_or_add_mag("2-2-1")

    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8), (10, 10, 10)
    )  # this is bigger than the bounding_box
    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (120, 24, 58)


def test_update_new_bounding_box_offset() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", LayerCategories.COLOR_TYPE).add_mag("1")

    assert ds.properties.data_layers["color"].bounding_box["topLeft"] == (-1, -1, -1)

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(
        write_data, offset=(10, 10, 10)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (10, 10, 10)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (10, 10, 10)

    mag.write(
        write_data, offset=(5, 5, 20)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (5, 5, 10)
    assert ds.properties.data_layers["color"].get_bounding_box_size() == (15, 15, 20)


def test_write_multi_channel_uint8() -> None:
    dataset_path = TESTOUTPUT_DIR / "multichannel"
    delete_dir(dataset_path)

    ds = Dataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer("color", LayerCategories.COLOR_TYPE, num_channels=3).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    ds.get_layer("color").get_mag("1").write(data)

    assert np.array_equal(data, mag.read(size=(250, 200, 10)))


def test_wk_write_multi_channel_uint16() -> None:
    dataset_path = TESTOUTPUT_DIR / "multichannel"
    delete_dir(dataset_path)

    ds = Dataset.create(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", LayerCategories.COLOR_TYPE, num_channels=3, dtype_per_layer="uint48"
    ).add_mag("1")

    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read(size=(250, 200, 10))

    assert np.array_equal(data, written_data)


def test_empty_read() -> None:
    filename = TESTOUTPUT_DIR / "empty_wk_dataset"
    delete_dir(filename)

    mag = (
        Dataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", LayerCategories.COLOR_TYPE)
        .add_mag("1")
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(offset=(0, 0, 0), size=(0, 0, 0))


def test_read_padded_data() -> None:
    filename = TESTOUTPUT_DIR / "empty_wk_dataset"
    delete_dir(filename)

    mag = (
        Dataset.create(filename, scale=(1, 1, 1))
        .add_layer("color", LayerCategories.COLOR_TYPE, num_channels=3)
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


def test_read_and_write_of_properties() -> None:
    destination_path = TESTOUTPUT_DIR / "read_write_properties"
    delete_dir(destination_path)
    source_file_name = TESTDATA_DIR / "simple_wk_dataset" / "datasource-properties.json"
    destination_file_name = destination_path / "datasource-properties.json"

    imported_properties = Properties._from_json(source_file_name)
    imported_properties._path = destination_file_name
    makedirs(destination_path)
    imported_properties._export_as_json()

    with open(source_file_name) as source_stream:
        source_data = json.load(source_stream)
        with open(destination_file_name) as destination_stream:
            destination_data = json.load(destination_stream)
            assert source_data == destination_data


def test_read_and_write_of_view_configuration() -> None:
    destination_path = TESTOUTPUT_DIR / "read_write_view_configuration"
    delete_dir(destination_path)
    source_file_name = TESTDATA_DIR / "simple_wk_dataset" / "datasource-properties.json"
    destination_file_name = destination_path / "datasource-properties.json"

    imported_properties = Properties._from_json(source_file_name)
    imported_properties._path = destination_file_name
    makedirs(destination_path)
    imported_properties._export_as_json()

    with open(source_file_name) as source_stream:
        source_data = json.load(source_stream)
        with open(destination_file_name) as destination_stream:
            destination_data = json.load(destination_stream)
            assert source_data == destination_data


def test_num_channel_mismatch_assertion() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    mag = ds.add_layer("color", LayerCategories.COLOR_TYPE, num_channels=1).add_mag(
        "1"
    )  # num_channel=1 is also the default

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)  # 3 channels

    with pytest.raises(AssertionError):
        mag.write(write_data)  # there is a mismatch between the number of channels


def test_get_or_add_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_layer="uint8", num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_layer="uint8", num_channels=1
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"

    with pytest.raises(AssertionError):
        # The layer "color" did exist before but with another 'dtype_per_layer' (this would work the same for 'category' and 'num_channels')
        ds.get_or_add_layer(
            "color",
            LayerCategories.COLOR_TYPE,
            dtype_per_layer="uint16",
            num_channels=1,
        )


def test_get_or_add_layer_idempotence() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")
    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1))
    ds.get_or_add_layer("color2", "color", np.uint8).get_or_add_mag("1")
    ds.get_or_add_layer("color2", "color", np.uint8).get_or_add_mag("1")


def test_get_or_add_mag() -> None:
    delete_dir(TESTOUTPUT_DIR / "wk_dataset")

    layer = Dataset.create(TESTOUTPUT_DIR / "wk_dataset", scale=(1, 1, 1)).add_layer(
        "color", LayerCategories.COLOR_TYPE
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


def test_open_dataset_without_num_channels_in_properties() -> None:
    delete_dir(TESTOUTPUT_DIR / "old_wk_dataset")
    copytree(TESTDATA_DIR / "old_wk_dataset", TESTOUTPUT_DIR / "old_wk_dataset")

    with open(
        TESTOUTPUT_DIR / "old_wk_dataset" / "datasource-properties.json"
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") is None

    ds = Dataset(TESTOUTPUT_DIR / "old_wk_dataset")
    assert ds.properties.data_layers["color"].num_channels == 1
    ds.properties._export_as_json()

    with open(
        TESTOUTPUT_DIR / "old_wk_dataset" / "datasource-properties.json"
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") == 1


def test_largest_segment_id_requirement() -> None:
    path = TESTOUTPUT_DIR / "largest_segment_id"
    delete_dir(path)
    ds = Dataset.create(path, scale=(10, 10, 10))

    with pytest.raises(AssertionError):
        ds.add_layer("segmentation", LayerCategories.SEGMENTATION_TYPE)

    largest_segment_id = 10
    ds.add_layer(
        "segmentation",
        LayerCategories.SEGMENTATION_TYPE,
        largest_segment_id=largest_segment_id,
    ).add_mag(Mag(1))

    ds = Dataset(path)
    assert (
        cast(
            SegmentationLayerProperties, ds.properties.data_layers["segmentation"]
        ).largest_segment_id
        == largest_segment_id
    )


def test_properties_with_segmentation() -> None:
    input_json_path = (
        TESTDATA_DIR / "complex_property_ds" / "datasource-properties.json"
    )
    output_json_path = (
        TESTOUTPUT_DIR / "complex_property_ds" / "datasource-properties.json"
    )
    properties = Properties._from_json(input_json_path)

    # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
    segmentation_layer = cast(
        SegmentationLayerProperties, properties.data_layers["segmentation"]
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


def test_chunking_wk(tmp_path: Path) -> None:
    ds = Dataset.create(Path(tmp_path), scale=(2, 2, 1))
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    mag = layer.add_mag("1", file_len=8, block_len=8)

    original_data = (np.random.rand(50, 100, 150) * 205).astype(np.uint8)
    mag.write(offset=(70, 80, 90), data=original_data)

    with get_executor_for_args(None) as executor:
        mag.for_each_chunk(
            chunk_job,
            chunk_size=(64, 64, 64),
            executor=executor,
        )

    assert np.array_equal(original_data + 50, mag.get_view().read()[0])


def test_chunking_wk_advanced() -> None:
    delete_dir(TESTOUTPUT_DIR / "chunking_dataset_wk_advanced")

    ds = Dataset.create(
        TESTOUTPUT_DIR / "chunking_dataset_wk_advanced", scale=(1, 1, 2)
    )
    mag = ds.add_layer(
        "color",
        category=LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1")
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view(size=(150, 150, 54), offset=(10, 10, 10))

    for_each_chunking_advanced(ds, view)


def test_chunking_wk_wrong_chunk_size() -> None:
    delete_dir(TESTOUTPUT_DIR / "chunking_dataset_wk_with_wrong_chunk_size")
    ds = Dataset.create(
        TESTOUTPUT_DIR / "chunking_dataset_wk_with_wrong_chunk_size", scale=(1, 1, 2)
    )
    mag = ds.add_layer(
        "color",
        category=LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1", block_len=8, file_len=8)
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view(size=(256, 256, 256))

    for_each_chunking_with_wrong_chunk_size(view)


def test_view_write_without_open() -> None:
    ds_path = TESTOUTPUT_DIR / "wk_dataset_write_without_open"
    delete_dir(ds_path)

    ds = Dataset.create(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", LayerCategories.COLOR_TYPE)
    layer.set_bounding_box(
        offset=(0, 0, 0), size=(64, 64, 64)
    )  # This newly created dataset would otherwise have a "empty" bounding box
    mag = layer.add_mag("1")

    wk_view = mag.get_view(size=(32, 64, 16))

    assert not wk_view._is_opened

    write_data = (np.random.rand(32, 64, 16) * 255).astype(np.uint8)
    wk_view.write(write_data)

    assert not wk_view._is_opened


def test_typing_of_get_mag() -> None:
    ds = Dataset(TESTDATA_DIR / "simple_wk_dataset")
    layer = ds.get_layer("color")
    assert layer.get_mag("1") == layer.get_mag(1)
    assert layer.get_mag("1") == layer.get_mag((1, 1, 1))
    assert layer.get_mag("1") == layer.get_mag([1, 1, 1])
    assert layer.get_mag("1") == layer.get_mag(np.array([1, 1, 1]))
    assert layer.get_mag("1") == layer.get_mag(Mag(1))


def test_dataset_get_or_create() -> None:
    ds_path = TESTOUTPUT_DIR / "wk_dataset_get_or_create"
    delete_dir(ds_path)

    # dataset does not exists yet
    ds1 = Dataset.get_or_create(ds_path, scale=(1, 1, 1))
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", LayerCategories.COLOR_TYPE)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = Dataset.get_or_create(ds_path, scale=(1, 1, 1))
    assert "color" in ds2.layers.keys()

    with pytest.raises(AssertionError):
        # dataset already exists, but with a different scale
        Dataset.get_or_create(ds_path, scale=(2, 2, 2))


def test_changing_layer_bounding_box() -> None:
    delete_dir(TESTOUTPUT_DIR / "test_changing_layer_bounding_box")
    copytree(
        TESTDATA_DIR / "simple_wk_dataset",
        TESTOUTPUT_DIR / "test_changing_layer_bounding_box",
    )

    ds = Dataset(TESTOUTPUT_DIR / "test_changing_layer_bounding_box")
    layer = ds.get_layer("color")
    mag = layer.get_mag("1")

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (24, 24, 24)
    original_data = mag.read(size=bbox_size)
    assert original_data.shape == (3, 24, 24, 24)

    layer.set_bounding_box_size((12, 12, 10))  # decrease bounding box

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (12, 12, 10)
    less_data = mag.read(size=bbox_size)
    assert less_data.shape == (3, 12, 12, 10)
    assert np.array_equal(original_data[:, :12, :12, :10], less_data)

    layer.set_bounding_box_size((36, 48, 60))  # increase the bounding box

    bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert bbox_size == (36, 48, 60)
    more_data = mag.read(size=bbox_size)
    assert more_data.shape == (3, 36, 48, 60)
    assert np.array_equal(more_data[:, :24, :24, :24], original_data)

    assert ds.properties.data_layers["color"].get_bounding_box_offset() == (0, 0, 0)

    # Move the offset from (0, 0, 0) to (10, 10, 0)
    # Note that the bottom right coordinate of the dataset is still at (24, 24, 24)
    layer.set_bounding_box(offset=(10, 10, 0), size=(14, 14, 24))

    new_bbox_offset = ds.properties.data_layers["color"].get_bounding_box_offset()
    new_bbox_size = ds.properties.data_layers["color"].get_bounding_box_size()
    assert new_bbox_offset == (10, 10, 0)
    assert new_bbox_size == (14, 14, 24)
    # Note that even though the offset was changed (in the properties), the offset of 'mag.read()'
    # still refers to the absolute position (relative to (0, 0, 0)).
    # The default offset is (0, 0, 0). Since the bottom right did not change, the read data equals 'original_data'.
    assert np.array_equal(original_data, mag.read())

    assert np.array_equal(
        original_data[:, 10:, 10:, :], mag.read(offset=(10, 10, 0), size=(14, 14, 24))
    )

    # resetting the offset to (0, 0, 0)
    # Note that the size did not change. Therefore, the new bottom right is now at (14, 14, 24)
    layer.set_bounding_box_offset((0, 0, 0))
    new_data = mag.read()
    assert new_data.shape == (3, 14, 14, 24)
    assert np.array_equal(original_data[:, :14, :14, :], new_data)


def test_get_view() -> None:
    delete_dir(TESTOUTPUT_DIR / "get_view_tests")

    ds = Dataset.create(TESTOUTPUT_DIR / "get_view_tests", scale=(1, 1, 1))
    mag = ds.add_layer("color", LayerCategories.COLOR_TYPE).add_mag("1")

    # The dataset is new -> no data has been written.
    # Therefore, the size of the bounding box in the properties.json is (0, 0, 0)

    # Creating this view works because the size is set to (0, 0, 0)
    # However, in practice a view with size (0, 0, 0) would not make sense
    with pytest.raises(AssertionError):
        # The offset and size default to (0, 0, 0).
        # Sizes that contain "0" are not allowed
        mag.get_view()

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


def test_adding_layer_with_invalid_dtype_per_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "invalid_dtype")

    ds = Dataset.create(TESTOUTPUT_DIR / "invalid_dtype", scale=(1, 1, 1))
    with pytest.raises(TypeError):
        # this would lead to a dtype_per_channel of "uint10", but that is not a valid dtype
        ds.add_layer(
            "color",
            LayerCategories.COLOR_TYPE,
            dtype_per_layer="uint30",
            num_channels=3,
        )
    with pytest.raises(TypeError):
        # "int" is interpreted as "int64", but 64 bit cannot be split into 3 channels
        ds.add_layer(
            "color", LayerCategories.COLOR_TYPE, dtype_per_layer="int", num_channels=3
        )
    ds.add_layer(
        "color", LayerCategories.COLOR_TYPE, dtype_per_layer="int", num_channels=4
    )  # "int"/"int64" works with 4 channels


def test_adding_layer_with_valid_dtype_per_layer() -> None:
    delete_dir(TESTOUTPUT_DIR / "valid_dtype")

    ds = Dataset.create(TESTOUTPUT_DIR / "valid_dtype", scale=(1, 1, 1))
    ds.add_layer(
        "color1", LayerCategories.COLOR_TYPE, dtype_per_layer="uint24", num_channels=3
    )
    ds.add_layer(
        "color2", LayerCategories.COLOR_TYPE, dtype_per_layer=np.uint8, num_channels=1
    )
    ds.add_layer(
        "color3", LayerCategories.COLOR_TYPE, dtype_per_channel=np.uint8, num_channels=3
    )
    ds.add_layer(
        "color4", LayerCategories.COLOR_TYPE, dtype_per_channel="uint8", num_channels=3
    )
    ds.add_layer(
        "seg1",
        LayerCategories.SEGMENTATION_TYPE,
        dtype_per_channel="float",
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg2",
        LayerCategories.SEGMENTATION_TYPE,
        dtype_per_channel=np.float,
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg3",
        LayerCategories.SEGMENTATION_TYPE,
        dtype_per_channel=float,
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg4",
        LayerCategories.SEGMENTATION_TYPE,
        dtype_per_channel="double",
        num_channels=1,
        largest_segment_id=100000,
    )
    ds.add_layer(
        "seg5",
        LayerCategories.SEGMENTATION_TYPE,
        dtype_per_channel="float",
        num_channels=3,
        largest_segment_id=100000,
    )

    with open(TESTOUTPUT_DIR / "valid_dtype" / "datasource-properties.json", "r") as f:
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

    ds = Dataset(
        TESTOUTPUT_DIR / "valid_dtype"
    )  # reopen the dataset to check if the data is read from the properties correctly
    assert ds.properties.data_layers["color1"].element_class == "uint24"
    assert ds.properties.data_layers["color2"].element_class == "uint8"
    assert ds.properties.data_layers["color3"].element_class == "uint24"
    assert ds.properties.data_layers["color4"].element_class == "uint24"
    # Note that 'float' and 'double' are stored as 'float32' and 'float64'
    assert ds.properties.data_layers["seg1"].element_class == "float32"
    assert ds.properties.data_layers["seg2"].element_class == "float32"
    assert ds.properties.data_layers["seg3"].element_class == "float32"
    assert ds.properties.data_layers["seg4"].element_class == "float64"
    assert ds.properties.data_layers["seg5"].element_class == "float96"


def test_writing_subset_of_compressed_data_multi_channel() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    write_data1 = (np.random.rand(3, 100, 120, 140) * 255).astype(np.uint8)
    Dataset.create(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1)).add_layer(
        "color", LayerCategories.COLOR_TYPE, num_channels=3
    ).add_mag("1", block_len=8, file_len=8).write(write_data1)

    # compress data
    compress_mag_inplace(
        (TESTOUTPUT_DIR / "compressed_data").resolve(),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        Dataset(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("1")
    )

    write_data2 = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
    compressed_mag.get_view(offset=(50, 60, 70)).write(
        offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
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
    Dataset.create(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1)).add_layer(
        "color", LayerCategories.COLOR_TYPE
    ).add_mag("1", block_len=8, file_len=8).write(write_data1)

    # compress data
    compress_mag_inplace(
        TESTOUTPUT_DIR / "compressed_data",
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        Dataset(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("1")
    )

    write_data2 = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
    compressed_mag.get_view(offset=(50, 60, 70)).write(
        offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
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
    Dataset.create(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1)).add_layer(
        "color", LayerCategories.COLOR_TYPE
    ).add_mag("1", block_len=8, file_len=8).write(
        (np.random.rand(20, 40, 60) * 255).astype(np.uint8)
    )

    # compress data
    compress_mag_inplace(
        (TESTOUTPUT_DIR / "compressed_data").resolve(),
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_mag = (
        Dataset(TESTOUTPUT_DIR / "compressed_data").get_layer("color").get_mag("1")
    )

    with pytest.raises(WKWException):
        # calling 'write' with unaligned data on compressed data without setting 'allow_compressed_write=True'
        compressed_mag.write(
            offset=(10, 20, 30),
            data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
        )


def test_writing_subset_of_chunked_compressed_data() -> None:
    delete_dir(TESTOUTPUT_DIR / "compressed_data")

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    Dataset.create(TESTOUTPUT_DIR / "compressed_data", scale=(1, 1, 1)).add_layer(
        "color", LayerCategories.COLOR_TYPE
    ).add_mag("1", block_len=8, file_len=8).write(write_data1)

    # compress data
    compress_mag_inplace(
        TESTOUTPUT_DIR / "compressed_data",
        layer_name="color",
        mag=Mag("1"),
    )

    # open compressed dataset
    compressed_view = (
        Dataset(TESTOUTPUT_DIR / "compressed_data")
        .get_layer("color")
        .get_mag("1")
        .get_view(size=(100, 200, 300))
    )

    # Easy case:
    # The aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
    write_data2 = (np.random.rand(50, 40, 30) * 255).astype(np.uint8)
    compressed_view.write(
        offset=(10, 20, 30), data=write_data2, allow_compressed_write=True
    )

    # Advanced case:
    # The aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
    compressed_view.write(
        offset=(10, 20, 30),
        data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
        allow_compressed_write=True,
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

    original_mag = (
        Dataset(TESTOUTPUT_DIR / "simple_wk_dataset_copy")
        .get_layer("color")
        .get_mag("1")
    )

    ds = Dataset.create(TESTOUTPUT_DIR / "wk_dataset_with_symlink", scale=(1, 1, 1))
    symlink_layer = ds.add_symlink_layer(
        TESTOUTPUT_DIR / "simple_wk_dataset_copy" / "color"
    )
    mag = symlink_layer.get_mag("1")

    assert (TESTOUTPUT_DIR / "wk_dataset_with_symlink" / "color" / "1").exists()

    assert len(ds.properties.data_layers) == 1
    assert len(ds.properties.data_layers["color"].wkw_magnifications) == 1

    # write data in symlink layer
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data)

    assert np.array_equal(mag.read(size=(10, 10, 10)), write_data)
    assert np.array_equal(original_mag.read(size=(10, 10, 10)), write_data)


def test_search_dataset_also_for_long_layer_name() -> None:
    delete_dir(TESTOUTPUT_DIR / "long_layer_name")

    ds = Dataset.create(TESTOUTPUT_DIR / "long_layer_name", scale=(1, 1, 1))
    mag = ds.add_layer("color", LayerCategories.COLOR_TYPE).add_mag("2")

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
    layer = Dataset(TESTOUTPUT_DIR / "long_layer_name").get_layer("color")
    mag = layer.get_mag("2")
    assert np.array_equal(
        mag.read(offset=(10, 10, 10), size=(10, 10, 10)), np.expand_dims(write_data, 0)
    )
    layer.delete_mag("2")


def test_outdated_dtype_parameter() -> None:
    delete_dir(TESTOUTPUT_DIR / "outdated_dtype")

    ds = Dataset.create(TESTOUTPUT_DIR / "outdated_dtype", scale=(1, 1, 1))
    with pytest.raises(ValueError):
        ds.get_or_add_layer(
            "color", LayerCategories.COLOR_TYPE, dtype=np.uint8, num_channels=1
        )

    with pytest.raises(ValueError):
        ds.add_layer(
            "color", LayerCategories.COLOR_TYPE, dtype=np.uint8, num_channels=1
        )


def test_dataset_conversion() -> None:
    origin_ds_path = TESTOUTPUT_DIR / "conversion" / "origin_wk"
    converted_ds_path = TESTOUTPUT_DIR / "conversion" / "converted_wk"

    delete_dir(origin_ds_path)
    delete_dir(converted_ds_path)

    # create example dataset
    origin_ds = Dataset.create(origin_ds_path, scale=(1, 1, 1))
    seg_layer = origin_ds.add_layer(
        "layer1",
        LayerCategories.SEGMENTATION_TYPE,
        num_channels=1,
        largest_segment_id=1000000000,
    )
    seg_layer.add_mag("1", block_len=8, file_len=16).write(
        offset=(10, 20, 30), data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8)
    )
    seg_layer.add_mag("2", block_len=8, file_len=16).write(
        offset=(5, 10, 15), data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8)
    )
    wk_color_layer = origin_ds.add_layer(
        "layer2", LayerCategories.COLOR_TYPE, num_channels=3
    )
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


def test_for_zipped_chunks() -> None:
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_source")
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_target")

    ds = Dataset.create(TESTOUTPUT_DIR / "zipped_chunking_source", scale=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=LayerCategories.COLOR_TYPE,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag("1")
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    source_view = mag.get_view(size=(256, 256, 256))

    target_mag = (
        Dataset.create(TESTOUTPUT_DIR / "zipped_chunking_target", scale=(1, 1, 2))
        .get_or_add_layer(
            "color",
            LayerCategories.COLOR_TYPE,
            dtype_per_channel="uint8",
            num_channels=3,
        )
        .get_or_add_mag("1", block_len=8, file_len=4)
    )

    target_mag.layer.set_bounding_box(offset=(0, 0, 0), size=(256, 256, 256))
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
    delete_dir(TESTOUTPUT_DIR / "zipped_chunking_source_invalid")

    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    ds = Dataset.create(
        TESTOUTPUT_DIR / "zipped_chunking_source_invalid", scale=(1, 1, 1)
    )
    layer1 = ds.get_or_add_layer("color1", LayerCategories.COLOR_TYPE)
    source_mag_view = layer1.get_or_add_mag(1, block_len=8, file_len=8)

    layer2 = ds.get_or_add_layer("color2", LayerCategories.COLOR_TYPE)
    target_mag_view = layer2.get_or_add_mag(1, block_len=8, file_len=8)

    source_view = source_mag_view.get_view(size=(300, 300, 300), read_only=True)
    # In this test case it is possible to simply set "read_only" for "target_view"
    # because the function "func" does not really write data to the target_view.
    # In a real scenario, calling "layer2.set_bounding_box(...)" and not setting "read_only" is recommended.
    target_view = target_mag_view.get_view(size=(300, 300, 300), read_only=True)

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


def test_read_only_view() -> None:
    delete_dir(TESTOUTPUT_DIR / "read_only_view")
    ds = Dataset.create(TESTOUTPUT_DIR / "read_only_view", scale=(1, 1, 1))
    mag = ds.get_or_add_layer("color", LayerCategories.COLOR_TYPE).get_or_add_mag("1")
    mag.write(
        data=(np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8), offset=(10, 20, 30)
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = (np.random.rand(1, 5, 6, 7) * 255).astype(np.uint8)
    with pytest.raises(AssertionError):
        v_read.write(data=new_data)

    v_write.write(data=new_data)


@pytest.fixture()
def create_dataset(tmp_path: Path) -> Generator[MagView, None, None]:
    ds = Dataset.create(Path(tmp_path), scale=(2, 2, 1))

    mag = ds.add_layer("color", "color").add_mag(
        "2-2-1", block_len=8, file_len=8
    )  # cube_size = 8*8 = 64
    yield mag


def test_bounding_box_on_disk(create_dataset: MagView) -> None:
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


def test_compression(tmp_path: Path) -> None:
    copytree(Path("testdata", "simple_wk_dataset"), tmp_path / "dataset")

    mag1 = Dataset(tmp_path / "dataset").get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, offset=(60, 80, 100))

    assert not mag1._is_compressed()
    mag1.compress()
    assert mag1._is_compressed()

    assert np.array_equal(
        write_data, mag1.read(offset=(60, 80, 100), size=(10, 20, 30))
    )

    with pytest.raises(wkw.WKWException):
        # writing unaligned data to a compressed dataset
        mag1.write((np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8))

    mag1.write(
        (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8),
        allow_compressed_write=True,
    )


def test_get_largest_segment_id(tmp_path: Path) -> None:
    ds = Dataset.create(tmp_path, scale=(1, 1, 1))

    segmentation_layer = cast(
        SegmentationLayer,
        ds.add_layer(
            "segmentation", LayerCategories.SEGMENTATION_TYPE, largest_segment_id=999
        ),
    )
    assert segmentation_layer.largest_segment_id == 999
    segmentation_layer.largest_segment_id = 123
    assert segmentation_layer.largest_segment_id == 123


def test_get_or_add_layer_by_type(tmp_path: Path) -> None:
    ds = Dataset.create(tmp_path, scale=(1, 1, 1))
    with pytest.raises(IndexError):
        ds.get_segmentation_layer()  # fails
    _ = ds.add_layer(
        "segmentation", LayerCategories.SEGMENTATION_TYPE, largest_segment_id=999
    )  # adds layer
    _ = ds.get_segmentation_layer()  # works
    _ = ds.add_layer(
        "different_segmentation",
        LayerCategories.SEGMENTATION_TYPE,
        largest_segment_id=999,
    )  # adds another layer
    with pytest.raises(IndexError):
        ds.get_segmentation_layer()  # fails

    with pytest.raises(IndexError):
        ds.get_color_layer()  # fails
    _ = ds.add_layer("color", LayerCategories.COLOR_TYPE)  # adds layer
    _ = ds.get_color_layer()  # works
    _ = ds.add_layer(
        "different_color", LayerCategories.COLOR_TYPE
    )  # adds another layer
    with pytest.raises(IndexError):
        ds.get_color_layer()  # fails


def test_dataset_name(tmp_path: Path) -> None:
    ds = Dataset.create(tmp_path / "some_name", scale=(1, 1, 1))
    assert ds.name == "some_name"
    ds.name = "other_name"
    assert ds.name == "other_name"

    ds2 = Dataset.create(
        tmp_path / "some_new_name", scale=(1, 1, 1), name="very important dataset"
    )
    assert ds2.name == "very important dataset"


def test_add_copy_layer(tmp_path: Path) -> None:
    ds = Dataset.create(tmp_path / "ds", scale=(2, 2, 1))

    # Create dataset to copy data from
    other_ds = Dataset.create(tmp_path / "other_ds", scale=(2, 2, 1))
    original_color_layer = other_ds.add_layer("color", LayerCategories.COLOR_TYPE)
    original_color_layer.add_mag(1).write(
        offset=(10, 20, 30), data=(np.random.rand(32, 64, 128) * 255).astype(np.uint8)
    )

    # Copies the "color" layer from a different dataset
    ds.add_copy_layer(tmp_path / "other_ds" / "color")
    assert len(ds.layers) == 1
    color_layer = ds.get_layer("color")
    assert color_layer.get_bounding_box() == BoundingBox(
        topleft=(10, 20, 30), size=(32, 64, 128)
    )
    assert color_layer.mags.keys() == original_color_layer.mags.keys()
    assert len(color_layer.mags.keys()) >= 1
    for mag in color_layer.mags.keys():
        np.array_equal(
            color_layer.get_mag(mag).read(), original_color_layer.get_mag(mag).read()
        )
        # Test if the copied layer contains actual data
        assert np.max(color_layer.get_mag(mag).read()) > 0

    with pytest.raises(IndexError):
        # The dataset already has a layer called "color".
        ds.add_copy_layer(tmp_path / "other_ds" / "color")

    # Test if the changes of the properties are persisted on disk by opening it again
    assert "color" in Dataset(tmp_path / "ds").layers.keys()

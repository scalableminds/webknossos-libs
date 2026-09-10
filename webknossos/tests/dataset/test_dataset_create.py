import importlib
import json
import os
from unittest import mock

import numpy as np
import pytest
from upath import UPath

from tests.constants import (
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS,
    DATA_FORMATS_AND_OUTPUT_PATHS,
    assure_exported_properties,
    copy_simple_dataset,
    prepare_dataset_path,
)
from webknossos.dataset import (
    Dataset,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.defaults import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_CHUNKS_PER_SHARD,
    DEFAULT_DATA_FORMAT,
    DEFAULT_SHARD_SHAPE,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
)
from webknossos.geometry import (
    Vec3Int,
)
from webknossos.utils import (
    copytree,
    rmtree,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_create_dataset_with_layer_and_mag(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.add_layer("color", "color", data_format=data_format)

    mag1 = ds.get_layer("color").add_mag("1")
    mag2 = ds.get_layer("color").add_mag("2-2-1")

    if data_format == DataFormat.WKW:
        assert (ds_path / "color" / "1" / "header.wkw").exists()
        assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()
    elif data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "1" / ".zarray").exists()
        assert (ds_path / "color" / "2-2-1" / ".zarray").exists()
    elif data_format == DataFormat.Zarr3:
        assert (ds_path / "color" / "1" / "zarr.json").exists()
        assert (ds_path / "color" / "2-2-1" / "zarr.json").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assert mag1.path == ds_path / "color" / "1"
    assert mag1._properties.path == "./color/1"
    assert mag2.path == ds_path / "color" / "2-2-1"
    assert mag2._properties.path == "./color/2-2-1"

    assure_exported_properties(ds)


def test_create_default_layer() -> None:
    ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)

    assert layer.data_format == DataFormat.Zarr3


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_create_default_mag(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag_view = layer.add_mag("1")

    assert layer.data_format == data_format
    assert mag_view.info.chunk_shape.xyz == DEFAULT_CHUNK_SHAPE
    if data_format == DataFormat.Zarr:
        assert mag_view.info.shard_shape.xyz == DEFAULT_CHUNK_SHAPE
        assert mag_view.info.chunks_per_shard.xyz == Vec3Int.full(1)
    else:
        assert mag_view.info.shard_shape.xyz == DEFAULT_SHARD_SHAPE
        assert mag_view.info.chunks_per_shard.xyz == DEFAULT_CHUNKS_PER_SHARD
    assert mag_view.info.bounding_box.size.c == 1
    assert mag_view.info.compression_mode == True


def test_shipped_default_shard_shape() -> None:
    """The test session shrinks the default shard shape (see conftest.py), so
    this asserts the shipped default and exercises it end to end."""
    defaults = importlib.import_module("webknossos.dataset.defaults")
    try:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WK_DEFAULT_CHUNKS_PER_SHARD", None)
            shipped = importlib.reload(defaults)
            shipped_shard_shape = shipped.DEFAULT_SHARD_SHAPE
            shipped_chunks_per_shard = shipped.DEFAULT_CHUNKS_PER_SHARD
    finally:
        importlib.reload(defaults)

    assert shipped_chunks_per_shard == Vec3Int.full(32)
    assert shipped_shard_shape == Vec3Int.full(1024)

    ds_path = prepare_dataset_path(DataFormat.Zarr3, TESTOUTPUT_DIR)
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr3)
        .add_mag("1", shard_shape=shipped_shard_shape)
    )
    assert mag.info.shard_shape.xyz == shipped_shard_shape
    assert mag.info.chunks_per_shard.xyz == shipped_chunks_per_shard

    data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint8)
    mag.write(data, absolute_offset=(60, 80, 100), allow_resize=True)
    np.testing.assert_array_equal(
        data, mag.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))[0]
    )


def test_dtype_per_channel() -> None:
    ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    with pytest.warns(DeprecationWarning):
        layer = ds.add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint16",
            num_channels=3,
            data_format=DataFormat.WKW,
        )
    assert layer.dtype == np.dtype("uint16")

    with pytest.warns(DeprecationWarning):
        layer2 = ds.get_or_add_layer(
            "color2",
            COLOR_CATEGORY,
            dtype_per_channel="float32",
            num_channels=3,
            data_format=DataFormat.WKW,
        )
    assert layer2.dtype == np.dtype("float32")

    with pytest.warns(DeprecationWarning):
        assert layer.dtype_per_channel == np.dtype("uint16")


def test_create_dataset_with_explicit_header_fields() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="uint16",
        num_channels=3,
        data_format=DataFormat.WKW,
    )

    ds.get_layer("color").add_mag("1", chunk_shape=64, shard_shape=4096)
    ds.get_layer("color").add_mag("2-2-1")

    assert (ds_path / "color" / "1" / "header.wkw").exists()
    assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assert ds.get_layer("color").dtype == np.dtype("uint16")
    assert ds.get_layer("color").num_channels == 3
    assert ds.get_layer("color")._properties.dtype == "uint16"
    assert ds.get_layer("color").get_mag(1).info.chunk_shape.xyz == Vec3Int.full(64)
    assert ds.get_layer("color").get_mag(1).info.shard_shape.xyz == Vec3Int.full(4096)
    assert ds.get_layer("color").get_mag(1).info.chunks_per_shard.xyz == Vec3Int.full(
        64
    )
    assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
    # defaults are used
    assert (
        ds.get_layer("color").get_mag("2-2-1").info.chunk_shape.xyz
        == DEFAULT_CHUNK_SHAPE
    )
    assert (
        ds.get_layer("color").get_mag("2-2-1").info.shard_shape.xyz
        == DEFAULT_SHARD_SHAPE
    )
    assert (
        ds.get_layer("color").get_mag("2-2-1").info.chunks_per_shard.xyz
        == DEFAULT_CHUNKS_PER_SHARD
    )
    assert (
        ds.get_layer("color").get_mag("2-2-1")._properties.cube_length
        == DEFAULT_SHARD_SHAPE.x
    )

    assure_exported_properties(ds)


def test_deprecated_chunks_per_shard() -> None:
    with pytest.warns(DeprecationWarning):
        ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)

        ds = Dataset(ds_path, voxel_size=(1, 1, 1))
        ds.add_layer(
            "color",
            COLOR_CATEGORY,
            dtype="uint16",
            num_channels=3,
            data_format=DataFormat.WKW,
        )

        ds.get_layer("color").add_mag("1", chunk_shape=64, chunks_per_shard=64)
        ds.get_layer("color").add_mag("2-2-1")

        assert (ds_path / "color" / "1" / "header.wkw").exists()
        assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()

        assert len(ds.layers) == 1
        assert len(ds.get_layer("color").mags) == 2

        assert ds.get_layer("color").dtype == np.dtype("uint16")
        assert ds.get_layer("color")._properties.bounding_box.size.c == 3
        assert ds.get_layer("color")._properties.dtype == "uint16"
        assert ds.get_layer("color").get_mag(1).info.chunk_shape.xyz == Vec3Int.full(64)
        assert ds.get_layer("color").get_mag(1).info.shard_shape.xyz == Vec3Int.full(
            4096
        )
        assert ds.get_layer("color").get_mag(
            1
        ).info.chunks_per_shard.xyz == Vec3Int.full(64)
        assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
        # defaults are used
        assert (
            ds.get_layer("color").get_mag("2-2-1").info.chunk_shape.xyz
            == DEFAULT_CHUNK_SHAPE
        )
        assert (
            ds.get_layer("color").get_mag("2-2-1").info.shard_shape.xyz
            == DEFAULT_SHARD_SHAPE
        )
        assert (
            ds.get_layer("color").get_mag("2-2-1").info.chunks_per_shard.xyz
            == DEFAULT_CHUNKS_PER_SHARD
        )
        assert (
            ds.get_layer("color").get_mag("2-2-1")._properties.cube_length
            == DEFAULT_SHARD_SHAPE.x
        )

        assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_open_dataset(data_format: DataFormat, output_path: UPath) -> None:
    new_dataset_path = copy_simple_dataset(data_format, output_path)
    ds = Dataset.open(new_dataset_path)

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1
    assert ds.get_layer("color").data_format == data_format


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_modify_existing_dataset(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds1 = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds1.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype="float",
        num_channels=1,
        data_format=data_format,
    )

    ds2 = Dataset.open(ds_path)

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype="uint8",
        largest_segment_id=100000,
        data_format=data_format,
    ).add_mag("1")

    assert (ds_path / "segmentation" / "1").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


def test_open_dataset_without_num_channels_in_properties() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "old_wkw")
    copytree(TESTDATA_DIR / "old_wkw_dataset", ds_path)

    data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    assert data["dataLayers"][0].get("num_channels") is None

    ds = Dataset.open(ds_path)
    assert ds.get_layer("color").num_channels == 1
    ds._save_dataset_properties()

    data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    assert data["dataLayers"][0].get("numChannels") == 1

    assure_exported_properties(ds)


def test_dataset_exist_ok() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "exist_ok")
    rmtree(ds_path)

    # dataset does not exists yet
    ds1 = Dataset(ds_path, voxel_size=(1, 1, 1), exist_ok=False)
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", COLOR_CATEGORY)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = Dataset(ds_path, voxel_size=(1, 1, 1), exist_ok=True)
    assert "color" in ds2.layers.keys()

    ds2 = Dataset(
        ds_path, voxel_size=(1, 1, 1), name="wkw_dataset_exist_ok", exist_ok=True
    )
    assert "color" in ds2.layers.keys()

    with pytest.raises(RuntimeError):
        # dataset already exists, but with a different voxel_size
        Dataset(ds_path, voxel_size=(2, 2, 2), exist_ok=True)

    with pytest.raises(RuntimeError):
        # dataset already exists, but with a different name
        Dataset(
            ds_path, voxel_size=(1, 1, 1), name="some different name", exist_ok=True
        )

    assure_exported_properties(ds1)


def test_dataset_name() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path / "some_name", voxel_size=(1, 1, 1))
    assert ds.name == "some_name"
    ds.name = "other_name"
    assert ds.name == "other_name"

    ds2 = Dataset(
        ds_path / "some_new_name", voxel_size=(1, 1, 1), name="very important dataset"
    )
    assert ds2.name == "very important dataset"

    assure_exported_properties(ds)


def test_dataset_open_wrong_path() -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.open("wrong_path")

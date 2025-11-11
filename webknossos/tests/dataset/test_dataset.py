import itertools
import json
import os
import pickle
from collections.abc import Iterator
from typing import cast
from unittest import mock

import numpy as np
import pytest
from cluster_tools import get_executor
from jsonschema import validate
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
    use_minio,
)
from webknossos import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    Dataset,
    LayerCategoryType,
    RemoteDataset,
    View,
)
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.defaults import DEFAULT_DATA_FORMAT
from webknossos.dataset.layer.view._array import Zarr3ArrayInfo, Zarr3Config
from webknossos.dataset_properties import (
    AttachmentDataFormat,
    DataFormat,
    DatasetProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
    SegmentationLayerProperties,
)
from webknossos.dataset_properties.structuring import get_dataset_converter
from webknossos.geometry import BoundingBox, Mag, Vec3Int, VecIntLike
from webknossos.utils import (
    copytree,
    dump_path,
    is_fs_path,
    is_remote_path,
    named_partial,
    rmtree,
    snake_to_camel_case,
)


@pytest.fixture(autouse=True, scope="module")
def start_minio() -> Iterator[None]:
    with use_minio():
        yield


DATA_FORMATS = [DataFormat.WKW, DataFormat.Zarr, DataFormat.Zarr3]
DATA_FORMATS_AND_OUTPUT_PATHS = [
    (DataFormat.WKW, TESTOUTPUT_DIR),
    (DataFormat.Zarr, TESTOUTPUT_DIR),
    (DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR),
    (DataFormat.Zarr3, TESTOUTPUT_DIR),
    (DataFormat.Zarr3, REMOTE_TESTOUTPUT_DIR),
]
OUTPUT_PATHS = [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR]


def copy_simple_dataset(
    data_format: DataFormat, output_path: UPath, suffix: str | None = None
) -> UPath:
    suffix = (f"_{suffix}") if suffix is not None else ""
    new_dataset_path = output_path / f"simple_{data_format}_dataset{suffix}"
    rmtree(new_dataset_path)
    copytree(
        TESTDATA_DIR / f"simple_{data_format}_dataset",
        new_dataset_path,
    )
    return new_dataset_path


def prepare_dataset_path(
    data_format: DataFormat, output_path: UPath, suffix: str | None = None
) -> UPath:
    suffix = (f"_{suffix}") if suffix is not None else ""
    new_dataset_path = output_path / f"{data_format}_dataset{suffix}"
    rmtree(new_dataset_path)
    return new_dataset_path


def chunk_job(args: tuple[View, int]) -> None:
    (view, _i) = args
    # increment the color value of each voxel
    data = view.read()
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    view.write(data)


def default_chunk_config(
    data_format: DataFormat, chunk_shape: int = 32
) -> tuple[Vec3Int, Vec3Int]:
    if data_format == DataFormat.Zarr:
        return (Vec3Int.full(chunk_shape * 8), Vec3Int.full(chunk_shape * 8))
    else:
        return (Vec3Int.full(chunk_shape), Vec3Int.full(chunk_shape * 8))


def advanced_chunk_job(args: tuple[View, int]) -> None:
    view, _i = args

    # write different data for each chunk (depending on the topleft of the chunk)
    data = view.read()
    data = np.ones(data.shape, dtype=np.dtype("uint8")) * (
        sum(view.bounding_box.topleft) % 256
    )
    view.write(data)


def for_each_chunking_with_wrong_chunk_shape(view: View) -> None:
    with get_executor("sequential") as executor:
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_shape=(0, 64, 64),
                executor=executor,
            )
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_shape=(16, 64, 64),
                executor=executor,
            )
        with pytest.raises(AssertionError):
            view.for_each_chunk(
                chunk_job,
                chunk_shape=(100, 64, 64),
                executor=executor,
            )


def for_each_chunking_advanced(ds: Dataset, view: View) -> None:
    with get_executor("sequential") as executor:
        view.for_each_chunk(
            advanced_chunk_job,
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
        chunk = (
            ds.get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=offset, size=size)
        )
        chunk_data = chunk.read()
        np.testing.assert_array_equal(
            np.ones(chunk_data.shape, dtype=np.dtype("uint8"))
            * (sum(chunk.bounding_box.topleft) % 256),
            chunk_data,
        )


def copy_and_transform_job(args: tuple[View, View, int], name: str, val: int) -> None:
    (source_view, target_view, _i) = args
    # This method simply takes the data from the source_view, transforms it and writes it to the target_view

    # These assertions are just to demonstrate how the passed parameters can be accessed inside this method
    assert name == "foo"
    assert val == 42

    # increment the color value of each voxel
    data = source_view.read()
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    target_view.write(data)


def get_multichanneled_data(dtype: type) -> np.ndarray:
    data: np.ndarray = np.zeros((3, 250, 200, 10), dtype=dtype)
    max_value = np.iinfo(dtype).max
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[0, i, j, h] = (i * 256) % max_value
                data[1, i, j, h] = (j * 256) % max_value
                data[2, i, j, h] = (100 * 256) % max_value
    return data


def assure_exported_properties(ds: Dataset) -> None:
    reopened_ds = Dataset.open(ds.path)
    assert ds._properties == reopened_ds._properties, (
        "The properties did not match after reopening the dataset. This might indicate that the properties were not exported after they were changed in memory."
    )


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


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
def test_ome_ngff_0_4_metadata(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr, output_path)
    ds = Dataset(ds_path, voxel_size=(11, 11, 28))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr)
    layer.add_mag("1")
    layer.add_mag("2-2-1")

    assert (ds_path / ".zgroup").exists()
    assert (ds_path / "color" / ".zgroup").exists()
    assert (ds_path / "color" / ".zattrs").exists()
    assert (ds_path / "color" / "1" / ".zarray").exists()
    assert (ds_path / "color" / "2-2-1" / ".zarray").exists()

    zattrs = json.loads((ds_path / "color" / ".zattrs").read_bytes())
    assert len(zattrs["multiscales"][0]["datasets"]) == 2
    assert zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ] == [
        1,
        11,
        11,
        28,
    ]
    assert zattrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0][
        "scale"
    ] == [
        1,
        22,
        22,
        28,
    ]

    validate(
        instance=zattrs,
        schema=json.loads(
            UPath(
                "https://ngff.openmicroscopy.org/0.4/schemas/image.schema"
            ).read_bytes()
        ),
    )


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
def test_ome_ngff_0_5_metadata(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(ds_path, voxel_size=(11, 11, 28))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr3)
    layer.add_mag("1")
    layer.add_mag("2-2-1")

    assert (ds_path / "zarr.json").exists()
    assert (ds_path / "color" / "zarr.json").exists()
    assert (ds_path / "color" / "1" / "zarr.json").exists()
    assert (ds_path / "color" / "2-2-1" / "zarr.json").exists()

    zattrs = json.loads((ds_path / "color" / "zarr.json").read_bytes())["attributes"]
    assert zattrs["ome"]["version"] == "0.5"
    assert len(zattrs["ome"]["multiscales"][0]["datasets"]) == 2
    assert zattrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"] == [
        1,
        11,
        11,
        28,
    ]
    assert zattrs["ome"]["multiscales"][0]["datasets"][1]["coordinateTransformations"][
        0
    ]["scale"] == [
        1,
        22,
        22,
        28,
    ]

    validate(
        instance=zattrs,
        schema=json.loads(
            UPath(
                "https://ngff.openmicroscopy.org/0.5/schemas/image.schema"
            ).read_bytes()
        ),
    )


def test_ome_ngff_0_5_metadata_symlink() -> None:
    def recursive_chmod(ds_path: UPath, mode: int) -> None:
        from pathlib import Path

        # See https://docs.python.org/3/library/os.html#os.chmod for how to use mode
        pathlib_path = Path(str(ds_path))
        os.chmod(pathlib_path, mode)
        for root, dirs, files in os.walk(pathlib_path):
            root_path = Path(root)
            for _dir in dirs:
                path = root_path / _dir
                os.chmod(path, mode)
            for file in files:
                path = root_path / file
                os.chmod(path, mode)

    ds_path = copy_simple_dataset(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "original")
    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    # remove write permissions
    recursive_chmod(ds_path, 0o555)
    try:
        ref_path = prepare_dataset_path(
            DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "with_refs"
        )
        ds = Dataset(ref_path, voxel_size=(1, 1, 1))

        ds.add_layer_as_ref(ds_path / "color")

    finally:
        # restore write permissions
        recursive_chmod(ds_path, 0o777)


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_mag_paths(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR)
    layer = Dataset(ds_path, voxel_size=(1, 1, 4)).add_layer(
        "color",
        COLOR_CATEGORY,
        bounding_box=BoundingBox((0, 0, 0), (32, 32, 32)),
        data_format=data_format,
    )
    mag1 = layer.add_mag("1")
    mag2 = layer.add_mag("2-2-1")

    assert mag1._properties.path == "./color/1"
    assert mag2._properties.path == "./color/2-2-1"


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
    assert mag_view.info.chunk_shape.xyz == Vec3Int.full(32)
    if data_format == DataFormat.Zarr:
        assert mag_view.info.shard_shape.xyz == Vec3Int.full(32)
        assert mag_view.info.chunks_per_shard.xyz == Vec3Int.full(1)
    else:
        assert mag_view.info.shard_shape.xyz == Vec3Int.full(1024)
        assert mag_view.info.chunks_per_shard.xyz == Vec3Int.full(32)
    assert mag_view.info.num_channels == 1
    assert mag_view.info.compression_mode == True


def test_create_dataset_with_explicit_header_fields() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint16",
        num_channels=3,
        data_format=DataFormat.WKW,
    )

    ds.get_layer("color").add_mag("1", chunk_shape=64, shard_shape=4096)
    ds.get_layer("color").add_mag("2-2-1")

    assert (ds_path / "color" / "1" / "header.wkw").exists()
    assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assert ds.get_layer("color").dtype_per_channel == np.dtype("uint16")
    assert ds.get_layer("color")._properties.element_class == "uint48"
    assert ds.get_layer("color").get_mag(1).info.chunk_shape.xyz == Vec3Int.full(64)
    assert ds.get_layer("color").get_mag(1).info.shard_shape.xyz == Vec3Int.full(4096)
    assert ds.get_layer("color").get_mag(1).info.chunks_per_shard.xyz == Vec3Int.full(
        64
    )
    assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
    assert ds.get_layer("color").get_mag("2-2-1").info.chunk_shape.xyz == Vec3Int.full(
        32
    )  # defaults are used
    assert ds.get_layer("color").get_mag("2-2-1").info.shard_shape.xyz == Vec3Int.full(
        1024
    )  # defaults are used
    assert ds.get_layer("color").get_mag(
        "2-2-1"
    ).info.chunks_per_shard.xyz == Vec3Int.full(32)  # defaults are used
    assert ds.get_layer("color").get_mag("2-2-1")._properties.cube_length == 32 * 32

    assure_exported_properties(ds)


def test_deprecated_chunks_per_shard() -> None:
    with pytest.warns(DeprecationWarning):
        ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)

        ds = Dataset(ds_path, voxel_size=(1, 1, 1))
        ds.add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint16",
            num_channels=3,
            data_format=DataFormat.WKW,
        )

        ds.get_layer("color").add_mag("1", chunk_shape=64, chunks_per_shard=64)
        ds.get_layer("color").add_mag("2-2-1")

        assert (ds_path / "color" / "1" / "header.wkw").exists()
        assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()

        assert len(ds.layers) == 1
        assert len(ds.get_layer("color").mags) == 2

        assert ds.get_layer("color").dtype_per_channel == np.dtype("uint16")
        assert ds.get_layer("color")._properties.element_class == "uint48"
        assert ds.get_layer("color").get_mag(1).info.chunk_shape.xyz == Vec3Int.full(64)
        assert ds.get_layer("color").get_mag(1).info.shard_shape.xyz == Vec3Int.full(
            4096
        )
        assert ds.get_layer("color").get_mag(
            1
        ).info.chunks_per_shard.xyz == Vec3Int.full(64)
        assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
        assert ds.get_layer("color").get_mag(
            "2-2-1"
        ).info.chunk_shape.xyz == Vec3Int.full(32)  # defaults are used
        assert ds.get_layer("color").get_mag(
            "2-2-1"
        ).info.shard_shape.xyz == Vec3Int.full(1024)  # defaults are used
        assert ds.get_layer("color").get_mag(
            "2-2-1"
        ).info.chunks_per_shard.xyz == Vec3Int.full(32)  # defaults are used
        assert ds.get_layer("color").get_mag("2-2-1")._properties.cube_length == 32 * 32

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
        dtype_per_channel="float",
        num_channels=1,
        data_format=data_format,
    )

    ds2 = Dataset.open(ds_path)

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype_per_channel="uint8",
        largest_segment_id=100000,
        data_format=data_format,
    ).add_mag("1")

    assert (ds_path / "segmentation" / "1").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_read(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel
    assert wk_view.info.data_format == data_format


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        wk_view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    assert wk_view.info.data_format == data_format

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

    wk_view.write(write_data, allow_unaligned=True)

    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_direct_zarr_access(output_path: UPath, data_format: DataFormat) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    np.random.seed(1234)

    # write: zarr, read: wk
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.get_zarr_array()[:, 0:10, 0:10, 0:10].write(write_data).result()
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    np.testing.assert_array_equal(data, write_data)

    # write: wk, read: zarr
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data, absolute_offset=(0, 0, 0), allow_unaligned=True)
    data = mag.get_zarr_array()[:, 0:10, 0:10, 0:10].read().result()
    np.testing.assert_array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_out_of_bounds(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(
        data_format, output_path, "view_dataset_out_of_bounds"
    )

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        view = (
            Dataset.open(ds_path)
            .get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
        )

    with pytest.raises(AssertionError):
        view.write(
            np.zeros((200, 200, 5), dtype=np.uint8)
        )  # this is bigger than the bounding_box


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_mag_view_write_out_of_bounds(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert mag_view.info.data_format == data_format

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8), allow_resize=True, allow_unaligned=True
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_mag_view_write_out_of_bounds_mag2(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    color_layer = ds.get_layer("color")
    mag_view = color_layer.get_or_add_mag("2-2-1", compress=False)

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(24, 24, 24)
    mag_view.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8),
        absolute_offset=(20, 20, 10),
        allow_resize=True,
        allow_unaligned=True,
    )  # this is bigger than the bounding_box
    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(120, 24, 58)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_allow_resize(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag("1")

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)

    # this should fail
    with pytest.raises(
        ValueError, match=".*does not fit in the layer's bounding box.*"
    ):
        mag.write(absolute_offset=(0, 0, 0), data=write_data)

    # this should go through
    mag.write(absolute_offset=(0, 0, 0), data=write_data, allow_resize=True)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (10, 10, 10))
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)).squeeze(0)
    np.testing.assert_array_equal(data, write_data)

    # override with same bbox
    mag.write(
        absolute_offset=(0, 0, 0),
        data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
    )

    # resize to larger bbox
    mag.write(
        absolute_offset=(10, 10, 10),
        data=(np.random.rand(5, 5, 5) * 255).astype(np.uint8),
        allow_resize=True,
        allow_unaligned=True,
    )
    assert layer.bounding_box == BoundingBox((0, 0, 0), (15, 15, 15))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_allow_unaligned(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "color",
        COLOR_CATEGORY,
        bounding_box=BoundingBox((0, 0, 0), (32, 32, 32)),
        data_format=data_format,
    )
    mag = layer.add_mag(
        "1",
        chunk_shape=(8, 8, 8),
        shard_shape=(8, 8, 8) if data_format == DataFormat.Zarr else (16, 16, 16),
    )

    np.random.seed(1234)
    write_data = (np.random.rand(4, 4, 4) * 255).astype(np.uint8)

    # this should fail
    with pytest.raises(ValueError, match=".*is not aligned with the shard shape.*"):
        mag.write(absolute_offset=(0, 0, 0), data=write_data)

    # this should go through
    mag.write(absolute_offset=(0, 0, 0), data=write_data, allow_unaligned=True)

    data = mag.read(absolute_offset=(0, 0, 0), size=(4, 4, 4)).squeeze(0)
    np.testing.assert_array_equal(data, write_data)

    # override a whole shard
    mag.write(
        absolute_offset=(16, 16, 16),
        data=(np.random.rand(16, 16, 16) * 255).astype(np.uint8),
    )

    # override multiple shards
    mag.write(
        absolute_offset=(16, 16, 0),
        data=(np.random.rand(16, 16, 32) * 255).astype(np.uint8),
    )

    # override the whole bbox
    mag.write(
        absolute_offset=(0, 0, 0),
        data=(np.random.rand(32, 32, 32) * 255).astype(np.uint8),
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_views_are_equal(data_format: DataFormat, output_path: UPath) -> None:
    np.random.seed(1234)
    data: np.ndarray = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)

    path_a = prepare_dataset_path(data_format, output_path / "a")
    path_b = prepare_dataset_path(data_format, output_path / "b")
    mag_a = (
        Dataset(path_a, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=data_format,
            bounding_box=BoundingBox((0, 0, 0), data.shape),
        )
        .get_or_add_mag("1")
    )
    mag_b = (
        Dataset(path_b, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=data_format,
            bounding_box=BoundingBox((0, 0, 0), data.shape),
        )
        .get_or_add_mag("1")
    )

    mag_a.write(data)
    mag_b.write(data)
    assert mag_a.content_is_equal(mag_b)

    data = data + 10
    mag_b.write(data)
    assert not mag_a.content_is_equal(mag_b)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_update_new_bounding_box_offset(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = color_layer.add_mag("1", compress=False)

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(
        write_data,
        absolute_offset=(10, 10, 10),
        allow_resize=True,
        allow_unaligned=True,
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(10, 10, 10)
    assert color_layer.bounding_box.size == Vec3Int(10, 10, 10)

    mag.write(
        write_data, absolute_offset=(5, 5, 20), allow_resize=True, allow_unaligned=True
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(5, 5, 10)
    assert color_layer.bounding_box.size == Vec3Int(15, 15, 20)

    assure_exported_properties(ds)


def test_chunked_compressed_write() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            data_format=DataFormat.WKW,
            bounding_box=BoundingBox(Vec3Int(1019, 1019, 1019), Vec3Int(10, 10, 10)),
        )
        .get_or_add_mag(
            "1",
            compress=True,
        )
    )

    np.random.seed(1234)
    data: np.ndarray = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)

    # write data in the bottom-right cornor of a shard so that other shards have to be written too
    mag.write(data, absolute_offset=mag.info.shard_shape - Vec3Int(5, 5, 5))

    assert (
        mag.get_view(
            absolute_offset=mag.info.shard_shape - Vec3Int(5, 5, 5),
            size=Vec3Int(10, 10, 10),
        ).read()
        == data
    ).all()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_write_multi_channel_uint8(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    mag.write(data, allow_resize=True)

    np.testing.assert_array_equal(data, mag.read())

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_wkw_write_multi_channel_uint16(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=3,
        dtype_per_channel="uint16",
        data_format=data_format,
    ).add_mag("1")

    data = get_multichanneled_data(np.uint16)

    mag.write(data, allow_resize=True)
    written_data = mag.read()

    np.testing.assert_array_equal(data, written_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_empty_read(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", category=COLOR_CATEGORY, data_format=data_format)
        .add_mag("1")
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(absolute_offset=(0, 0, 0), size=(0, 0, 0))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("absolute_offset", [None, Vec3Int(12, 12, 12)])
def test_write_layer(
    data_format: DataFormat, output_path: UPath, absolute_offset: Vec3Int | None
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    np.random.seed(1234)
    data: np.ndarray = (np.random.rand(128, 128, 128) * 255).astype(np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        absolute_offset=absolute_offset,
    )

    np.testing.assert_array_equal(layer.get_mag(1).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft_xyz == absolute_offset
    assert layer.bounding_box.size_xyz == Vec3Int(data.shape)
    assert Mag(2) in layer.mags  # did downsample
    assert Mag(4) in layer.mags  # did downsample


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("absolute_offset", [None, Vec3Int(12, 12, 12)])
def test_write_layer_mag2(
    data_format: DataFormat, output_path: UPath, absolute_offset: Vec3Int | None
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(12, 12, 24))

    np.random.seed(1234)
    data: np.ndarray = (np.random.rand(128, 128, 128) * 255).astype(np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        absolute_offset=absolute_offset,
        mag=(2, 2, 1),
    )

    np.testing.assert_array_equal(layer.get_mag((2, 2, 1)).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft_xyz == absolute_offset  # in mag1
    assert layer.bounding_box.size_xyz == Vec3Int(data.shape) * Vec3Int(
        2, 2, 1
    )  # in mag1
    assert Mag((4, 4, 2)) in layer.mags  # did downsample


@pytest.mark.parametrize(
    "data_format,output_path",
    [(DataFormat.Zarr3, TESTOUTPUT_DIR), (DataFormat.Zarr3, REMOTE_TESTOUTPUT_DIR)],
)
@pytest.mark.parametrize("absolute_offset", [None, (3, 12, 12, 12)])
def test_write_layer_5d(
    data_format: DataFormat,
    output_path: UPath,
    absolute_offset: VecIntLike | None,
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    np.random.seed(1234)
    data: np.ndarray = (np.random.rand(3, 2, 128, 128, 128) * 255).astype(np.uint8)
    layer = ds.write_layer(
        "color",
        category=COLOR_CATEGORY,
        data=data,
        data_format=data_format,
        axes=("c", "t", "x", "y", "z"),
        shard_shape=(128, 128, 128),
        absolute_offset=absolute_offset,
    )

    np.testing.assert_array_equal(layer.get_mag(1).read().squeeze(), data)
    if absolute_offset is not None:
        assert layer.bounding_box.topleft.to_tuple() == absolute_offset
    assert layer.bounding_box.size.to_tuple() == data.shape[1:]


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_padded_data(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer(
            "color", category=COLOR_CATEGORY, num_channels=3, data_format=data_format
        )
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    np.testing.assert_array_equal(data, np.zeros((3, 10, 10, 10)))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_num_channel_mismatch_assertion(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer(
        "color", category=COLOR_CATEGORY, num_channels=1, data_format=data_format
    ).add_mag("1")  # num_channel=1 is also the default

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)  # 3 channels

    with pytest.raises(AssertionError):
        mag.write(
            write_data, allow_resize=True
        )  # there is a mismatch between the number of channels

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=1,
        data_format=data_format,
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"
    assert layer.data_format == data_format

    # layer did exist before
    layer = ds.get_or_add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=1,
        data_format=data_format,
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"
    assert layer.data_format == data_format

    with pytest.raises(AssertionError):
        # The layer "color" did exist before but with another dtype (this would work the same for 'category' and 'num_channels')
        ds.get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint16",
            num_channels=1,
            data_format=data_format,
        )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer_idempotence(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds.get_or_add_layer(
        "color2", category="color", dtype_per_channel=np.uint8, data_format=data_format
    ).get_or_add_mag("1")
    ds.get_or_add_layer(
        "color2", category="color", dtype_per_channel=np.uint8, data_format=data_format
    ).get_or_add_mag("1")

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)

    layer = Dataset(ds_path, voxel_size=(1, 1, 1)).add_layer(
        "color", category=COLOR_CATEGORY, data_format=data_format
    )

    assert Mag(1) not in layer.mags.keys()

    chunk_shape, shard_shape = default_chunk_config(data_format, 32)

    # The mag did not exist before
    mag = layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=True,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    # The mag did exist before
    layer.get_or_add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=True,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    with pytest.raises(ValueError):
        # The mag "1" did exist before but with another 'chunk_shape' (this would work the same for 'shard_shape' and 'compress')
        layer.get_or_add_mag(
            "1",
            chunk_shape=Vec3Int.full(64),
            shard_shape=shard_shape,
            compress=True,
        )

    assure_exported_properties(layer.dataset)


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


@pytest.mark.use_proxay
def test_explore_and_add_remote() -> None:
    remote_ds = RemoteDataset.explore_and_add_remote(
        "http://localhost:9000/data/v9/zarr/Organization_X/l4_sample/",
        "added_remote_ds",
        "/Organization_X",
    )
    assert remote_ds.name == "added_remote_ds"


def test_no_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(10, 10, 10))

    ds.add_layer("segmentation", SEGMENTATION_CATEGORY).add_mag(Mag(1))

    ds = Dataset.open(ds_path)

    assert (
        ds.get_layer("segmentation").as_segmentation_layer().largest_segment_id is None
    )

    assure_exported_properties(ds)


def test_properties_with_segmentation() -> None:
    ds_path = prepare_dataset_path(
        DataFormat.WKW, TESTOUTPUT_DIR, "complex_property_ds"
    )
    copytree(TESTDATA_DIR / "complex_property_ds", ds_path)

    data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    ds_properties = get_dataset_converter().structure(data, DatasetProperties)

    # the attributes 'largest_segment_id' and 'mappings' only exist if it is a SegmentationLayer
    segmentation_layer = cast(
        SegmentationLayerProperties,
        [layer for layer in ds_properties.data_layers if layer.name == "segmentation"][
            0
        ],
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

    # Update the properties on disk (without changing the data)
    (ds_path / PROPERTIES_FILE_NAME).write_text(
        json.dumps(
            get_dataset_converter().unstructure(ds_properties),
            indent=4,
        )
    )

    # validate if contents match
    input_data = json.loads(
        (TESTDATA_DIR / "complex_property_ds" / PROPERTIES_FILE_NAME).read_text()
    )

    output_data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
    for layer in output_data["dataLayers"]:
        # remove the num_channels because they are not part of the original json
        if "numChannels" in layer:
            del layer["numChannels"]

    assert input_data == output_data


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_relative_mag_paths(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)

    ds = Dataset.open(ds_path)
    for layer in ds.layers.values():
        for mag in layer.mags.values():
            if is_fs_path(mag.path):
                mag._properties.path = f"../{ds_path.name}/{layer.name}/{mag.path.name}"
            else:
                mag._properties.path = f"{layer.name}/{mag.path.name}"

    ds._save_dataset_properties()

    ds = Dataset.open(ds_path)
    for layer in ds.layers.values():
        for mag in layer.mags.values():
            assert mag.path == ds_path / layer.name / mag.path.name


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wk(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(
        "1",
        shard_shape=shard_shape,
        chunk_shape=chunk_shape,
    )

    original_data = (np.random.rand(50, 100, 150) * 205).astype(np.uint8)
    mag.write(absolute_offset=(70, 80, 90), data=original_data, allow_resize=True)

    # Test with executor
    with get_executor("sequential") as executor:
        mag.for_each_chunk(
            chunk_job,
            chunk_shape=shard_shape,
            executor=executor,
        )
    np.testing.assert_array_equal(original_data + 50, mag.get_view().read()[0])

    # Reset the data
    mag.write(absolute_offset=(70, 80, 90), data=original_data, allow_resize=True)

    # Test without executor
    mag.for_each_chunk(
        chunk_job,
        chunk_shape=shard_shape,
    )
    np.testing.assert_array_equal(original_data + 50, mag.get_view().read()[0])

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format", [DataFormat.WKW, DataFormat.Zarr3])
def test_chunking_wkw_advanced(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "chunking_advanced")
    ds = Dataset(ds_path, voxel_size=(1, 1, 2))

    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag(
        "1",
        chunk_shape=8,
        shard_shape=64,
    )
    mag.write(
        data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8),
        allow_resize=True,
    )
    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        view = mag.get_view(absolute_offset=(10, 10, 10), size=(150, 150, 54))
        for_each_chunking_advanced(ds, view)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wkw_wrong_chunk_shape(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "chunking_with_wrong_chunk_shape"
    )
    ds = Dataset(ds_path, voxel_size=(1, 1, 2))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag(
        "1",
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )
    mag.write(
        data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8),
        allow_resize=True,
    )
    view = mag.get_view()

    for_each_chunking_with_wrong_chunk_shape(view)

    assure_exported_properties(ds)


def test_typing_of_get_mag() -> None:
    ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")
    layer = ds.get_layer("color")
    assert layer.get_mag("1") == layer.get_mag(1)
    assert layer.get_mag("1") == layer.get_mag((1, 1, 1))
    assert layer.get_mag("1") == layer.get_mag([1, 1, 1])
    assert layer.get_mag("1") == layer.get_mag(np.array([1, 1, 1]))
    assert layer.get_mag("1") == layer.get_mag(Mag(1))

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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_changing_layer_bounding_box(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "change_layer_bounding_box")
    ds = Dataset.open(ds_path)
    layer = ds.get_layer("color")
    mag = layer.get_mag("1")

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (24, 24, 24)
    original_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert original_data.shape == (3, 24, 24, 24)

    layer.bounding_box = layer.bounding_box.with_size(
        [12, 12, 10]
    )  # decrease bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (12, 12, 10)
    less_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert less_data.shape == (3, 12, 12, 10)
    np.testing.assert_array_equal(original_data[:, :12, :12, :10], less_data)

    layer.bounding_box = layer.bounding_box.with_size(
        [36, 48, 60]
    )  # increase the bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (36, 48, 60)
    more_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
    assert more_data.shape == (3, 36, 48, 60)
    np.testing.assert_array_equal(more_data[:, :24, :24, :24], original_data)

    assert tuple(ds.get_layer("color").bounding_box.topleft) == (0, 0, 0)

    # Move the offset from (0, 0, 0) to (10, 10, 0)
    # Note that the bottom right coordinate of the dataset is still at (24, 24, 24)
    layer.bounding_box = BoundingBox((10, 10, 0), (14, 14, 24))

    new_bbox_offset = ds.get_layer("color").bounding_box.topleft
    new_bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(new_bbox_offset) == (10, 10, 0)
    assert tuple(new_bbox_size) == (14, 14, 24)
    np.testing.assert_array_equal(
        original_data,
        mag.read(absolute_offset=(0, 0, 0), size=mag.bounding_box.bottomright),
    )

    np.testing.assert_array_equal(
        original_data[:, 10:, 10:, :],
        mag.read(absolute_offset=(10, 10, 0), size=(14, 14, 24)),
    )

    # resetting the offset to (0, 0, 0)
    # Note that the size did not change. Therefore, the new bottom right is now at (14, 14, 24)
    layer.bounding_box = BoundingBox((0, 0, 0), new_bbox_size)
    new_data = mag.read()
    assert new_data.shape == (3, 14, 14, 24)
    np.testing.assert_array_equal(original_data[:, :14, :14, :], new_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_dataset_bounding_box_calculation(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "change_layer_bounding_box")
    ds = Dataset.open(ds_path)
    layer = ds.get_layer("color")
    # BoundingBox(topleft=(0, 0, 0), size=(24, 24, 24))
    assert layer.bounding_box == ds.calculate_bounding_box(), (
        "The calculated bounding box of the dataset does not "
        + "match the color layer's bounding box."
    )
    layer.bounding_box = layer.bounding_box.with_size((512, 512, 512))
    assert layer.bounding_box == ds.calculate_bounding_box(), (
        "The calculated bounding box of the dataset does not "
        + "match the color layer's enlarged bounding box."
    )


def test_get_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "get_view")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY).add_mag("1")

    # The dataset is new -> no data has been written.
    # Therefore, the size of the bounding box in the properties.json is (0, 0, 0)

    # Creating this view works because the size is set to (0, 0, 0)
    # However, in practice a view with size (0, 0, 0) would not make sense
    # Sizes that contain "0" are not allowed usually, except for an empty layer
    assert mag.get_view().bounding_box.is_empty()

    with pytest.raises(AssertionError):
        # This view exceeds the bounding box
        mag.get_view(relative_offset=(0, 0, 0), size=(16, 16, 16))

    # read-only-views may exceed the bounding box
    read_only_view = mag.get_view(
        relative_offset=(0, 0, 0), size=(16, 16, 16), read_only=True
    )
    assert read_only_view.bounding_box == BoundingBox((0, 0, 0), (16, 16, 16))

    with pytest.raises(AssertionError):
        # Trying to get a writable sub-view of a read-only-view is not allowed
        read_only_view.get_view(read_only=False)

    np.random.seed(1234)
    write_data = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    # This operation updates the bounding box of the dataset according to the written data
    mag.write(write_data, absolute_offset=(10, 20, 30), allow_resize=True)

    with pytest.raises(AssertionError):
        # The offset and size default to (0, 0, 0).
        # Sizes that contain "0" are not allowed
        mag.get_view(absolute_offset=(0, 0, 0), size=(10, 10, 0))

    assert mag.bounding_box.bottomright == Vec3Int(110, 220, 330)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        # Therefore, creating a view with a size of (16, 16, 16) is now allowed
        wk_view = mag.get_view(relative_offset=(0, 0, 0), size=(16, 16, 16))
    assert wk_view.bounding_box == BoundingBox((10, 20, 30), (16, 16, 16))

    with pytest.raises(AssertionError):
        # Creating this view does not work because the offset (0, 0, 0) would be outside
        # of the bounding box from the properties.json.
        mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0))

    # But setting "read_only=True" still works
    mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0), read_only=True)

    with pytest.warns(UserWarning, match=".*not aligned with the shard shape.*"):
        # Creating this subview works because the subview is completely inside the 'wk_view'.
        # Note that the offset in "get_view" is always relative to the "global_offset"-attribute of the called view.
        sub_view = wk_view.get_view(relative_offset=(8, 8, 8), size=(8, 8, 8))
    assert sub_view.bounding_box == BoundingBox((18, 28, 38), (8, 8, 8))

    with pytest.raises(AssertionError):
        # Creating this subview does not work because it is not completely inside the 'wk_view'
        wk_view.get_view(relative_offset=(8, 8, 8), size=(10, 10, 10))

    # Again: read-only is allowed
    wk_view.get_view(relative_offset=(8, 8, 8), size=(10, 10, 10), read_only=True)

    with pytest.raises(AssertionError):
        # negative offsets are not allowed
        mag.get_view(absolute_offset=(-1, -2, -3))

    assure_exported_properties(ds)


def test_adding_layer_with_invalid_dtype_per_layer() -> None:
    with pytest.warns(DeprecationWarning):
        ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "invalid_dtype")
        ds = Dataset(ds_path, voxel_size=(1, 1, 1))
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
    with pytest.warns(DeprecationWarning):
        ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "valid_dtype")
        ds = Dataset(ds_path, voxel_size=(1, 1, 1))
        ds.add_layer("color1", COLOR_CATEGORY, dtype_per_layer="uint24", num_channels=3)
        ds.add_layer("color2", COLOR_CATEGORY, dtype_per_layer=np.uint8, num_channels=1)
        ds.add_layer(
            "color3", COLOR_CATEGORY, dtype_per_channel=np.uint8, num_channels=3
        )
        ds.add_layer(
            "color4", COLOR_CATEGORY, dtype_per_channel="uint8", num_channels=3
        )

        data = json.loads((ds_path / PROPERTIES_FILE_NAME).read_text())
        # The order of the layers in the properties equals the order of creation
        assert data["dataLayers"][0]["elementClass"] == "uint24"
        assert data["dataLayers"][1]["elementClass"] == "uint8"
        assert data["dataLayers"][2]["elementClass"] == "uint24"
        assert data["dataLayers"][3]["elementClass"] == "uint24"

        reopened_ds = Dataset.open(
            ds_path
        )  # reopen the dataset to check if the data is read from the properties correctly
        assert reopened_ds.get_layer("color1").dtype_per_layer == "uint24"
        assert reopened_ds.get_layer("color2").dtype_per_layer == "uint8"
        assert reopened_ds.get_layer("color3").dtype_per_layer == "uint24"
        assert reopened_ds.get_layer("color4").dtype_per_layer == "uint24"

        assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_multi_channel(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = (np.random.rand(3, 100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True, allow_unaligned=True)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    write_data2 = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    # Writing unaligned data to a compressed dataset works because the data gets
    # padded, but it requires an explicit allow_unaligned=True flag
    # Writing compressed data directly to "compressed_mag" also works, but using a
    # View here covers an additional edge case
    with pytest.warns(UserWarning):
        view = compressed_mag.get_view(relative_offset=(50, 60, 70), size=(50, 60, 70))
    with pytest.raises(ValueError):
        view.write(relative_offset=(10, 20, 30), data=write_data2)
    view.write(relative_offset=(10, 20, 30), data=write_data2, allow_unaligned=True)

    np.testing.assert_array_equal(
        write_data2,
        compressed_mag.read(relative_offset=(60, 80, 100), size=(10, 10, 10)),
    )  # the new data was written
    np.testing.assert_array_equal(
        write_data1[:, :60, :80, :100],
        compressed_mag.read(relative_offset=(0, 0, 0), size=(60, 80, 100)),
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_single_channel(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    write_data2 = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)

    # Writing unaligned data to a compressed dataset works because the data gets
    # padded, but it requires an explicit allow_unaligned=True flag
    # Writing compressed data directly to "compressed_mag" also works, but using a
    # View here covers an additional edge case
    with pytest.warns(UserWarning):
        view = compressed_mag.get_view(absolute_offset=(50, 60, 70), size=(50, 60, 70))
    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        view.write(relative_offset=(10, 20, 30), data=write_data2)
    view.write(relative_offset=(10, 20, 30), data=write_data2, allow_unaligned=True)

    np.testing.assert_array_equal(
        write_data2,
        compressed_mag.read(absolute_offset=(60, 80, 100), size=(10, 10, 10))[0],
    )  # the new data was written
    np.testing.assert_array_equal(
        write_data1[:60, :80, :100],
        compressed_mag.read(absolute_offset=(0, 0, 0), size=(60, 80, 100))[0],
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "2",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(
        (np.random.rand(120, 140, 160) * 255).astype(np.uint8), allow_resize=True
    )

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("2")

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        compressed_mag.write(
            absolute_offset=(10, 20, 30),
            data=(np.random.rand(10, 10, 10) * 255).astype(np.uint8),
        )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_chunked_compressed_data(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)

    write_data1 = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    write_data2 = (np.random.rand(50, 40, 30) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1",
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=True,
        )
    )
    mag_view.write(write_data1, allow_resize=True, allow_unaligned=True)

    # open compressed dataset
    compressed_view = (
        Dataset.open(ds_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(100, 200, 300))
    )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        # Easy case:
        # The aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
        compressed_view.write(absolute_offset=(10, 20, 30), data=write_data2)
    compressed_view.write(
        absolute_offset=(10, 20, 30), data=write_data2, allow_unaligned=True
    )

    with pytest.raises(ValueError, match=".*not aligned with the shard shape.*"):
        # Advanced case:
        # The aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
        compressed_view.write(
            absolute_offset=(10, 20, 30),
            data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
        )
    compressed_view.write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
        allow_unaligned=True,
    )

    np.array_equal(
        write_data2,
        compressed_view.read(absolute_offset=(10, 20, 30), size=(50, 40, 30)),
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_view.read(absolute_offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
@pytest.mark.parametrize("as_object", [True, False])
def test_add_layer_as_ref(
    data_format: DataFormat, output_path: UPath, as_object: bool
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_refs")

    # Add an additional segmentation layer to the original dataset
    original_ds = Dataset.open(ds_path)
    original_ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    original_mag = original_ds.get_layer("color").get_mag("1")
    original_mag.write(
        (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8), allow_unaligned=True
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    # add color layer
    new_layer = ds.add_layer_as_ref(
        original_ds.get_layer("color") if as_object else ds_path / "color"
    )
    mag = new_layer.get_mag("1")
    # add segmentation layer
    new_segmentation_layer = ds.add_layer_as_ref(
        original_ds.get_layer("segmentation")
        if as_object
        else ds_path / "segmentation",
        new_layer_name="seg",
    )

    color_mag_path = original_mag.path.name
    assert ds._properties.data_layers[0].mags[0].path == dump_path(
        ds_path / "color" / color_mag_path, new_path
    )
    assert not (new_path / "color" / color_mag_path).exists()
    assert ds._properties.data_layers[1].mags[0].path == dump_path(
        ds_path / "segmentation" / "1", new_path
    )
    assert not (new_path / "segmentation" / "1").exists()
    assert not (new_path / "segmentation").exists()
    assert not (new_path / "seg" / "1").exists()
    assert not (new_path / "seg").exists()

    assert len(ds.layers) == 2
    assert len(ds.get_layer("color").mags) == 1

    assert new_segmentation_layer.as_segmentation_layer().largest_segment_id == 999

    assert not new_layer.read_only
    assert not new_segmentation_layer.read_only
    assert mag.read_only

    with pytest.raises(RuntimeError):
        mag.write(
            (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8), allow_unaligned=True
        )

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)),
        original_mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)),
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_add_layer_as_ref_prefix(output_path: UPath) -> None:
    source = Dataset(output_path / "name_with_suffix", (1, 1, 1))
    source.add_layer(
        "consensus", SEGMENTATION_CATEGORY, dtype_per_channel="uint8"
    ).add_mag(1)

    target = Dataset(output_path / "name", (1, 1, 1))
    target.add_layer("raw", COLOR_CATEGORY, dtype_per_channel="uint8").add_mag(1)

    glom = source.get_layer("consensus")
    target.add_layer_as_ref(foreign_layer=glom, new_layer_name="glomeruli")

    assert target._properties.data_layers[1].mags[0].path == dump_path(
        source.get_layer("consensus").get_mag(1).path,
        UPath.home() / "random",  # an unrelated path
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_ref_layer_add_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_refs")

    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    new_layer = ds.add_layer_as_ref(ds_path / "color")

    new_layer.add_mag(2)
    assert new_layer.get_mag(2).path == new_path / "color" / "2"
    assert new_layer.get_mag(2).path.exists()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_ref_layer_rename(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).add_mag(1)

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    ref_layer = ds.add_layer_as_ref(ds_path / "color")

    assert not (new_path / "color").exists()
    if is_fs_path(new_path):
        ref_layer.name = "color2"

        with pytest.raises(ValueError):
            ref_layer.name = "color/2"  # invalid name
    else:
        with pytest.raises(RuntimeError):
            ref_layer.name = "color2"


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_ref(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    original_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        bounding_box=BoundingBox((0, 0, 0), (10, 20, 30)),
    )
    original_layer.add_mag(1).write(
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )
    original_mag_2 = original_layer.add_mag(2)
    original_mag_2.write(data=(np.random.rand(5, 10, 15) * 255).astype(np.uint8))
    original_mag_4 = original_layer.add_mag(4)
    original_mag_4.write(data=(np.random.rand(3, 5, 8) * 255).astype(np.uint8))

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    layer.add_mag(1).write(
        absolute_offset=(6, 6, 6),
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8),
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    ref_mag_2 = layer.add_mag_as_ref(original_mag_2)
    ref_mag_4 = layer.add_mag_as_ref(ds_path / "color" / "4")
    assert ref_mag_2._properties.path == dump_path(ds_path / "color" / "2", new_path)
    assert ref_mag_4._properties.path == dump_path(ds_path / "color" / "4", new_path)

    assert (new_path / "color" / "1").exists()
    assert not (new_path / "color" / "2").exists()
    assert not (new_path / "color" / "4").exists()
    assert len(layer._properties.mags) == 3

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    assert not layer.read_only
    assert not layer.get_mag(1).read_only
    assert ref_mag_2.read_only

    np.testing.assert_array_equal(
        ref_mag_2.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0],
        original_layer.get_mag(2).read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0],
    )

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)

    layer.delete_mag(4)
    assert Mag(4) not in layer.mags
    assert not (new_path / "color" / "4").exists()
    assert (ds_path / "color" / "4").exists()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_ref_with_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    new_path = prepare_dataset_path(data_format, output_path, "with_ref")

    original_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        bounding_box=BoundingBox((0, 0, 0), (10, 20, 30)),
    )
    original_layer.add_mag(1).write(
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )

    ds = Dataset(new_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    layer.add_mag_as_ref(original_layer.get_mag(1), mag="2")

    assert list(layer.mags.values())[0].mag == Mag("2")
    assert list(layer.mags.values())[0]._properties.path == dump_path(
        ds_path / "color" / "1", new_path
    )

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_remote_add_symlink_layer(data_format: DataFormat) -> None:
    src_dataset_path = copy_simple_dataset(data_format, REMOTE_TESTOUTPUT_DIR)
    dst_dataset_path = prepare_dataset_path(
        data_format, REMOTE_TESTOUTPUT_DIR, "with_symlink"
    )

    src_ds = Dataset.open(src_dataset_path)
    dst_ds = Dataset(dst_dataset_path, voxel_size=(1, 1, 1))

    with pytest.raises(AssertionError):
        dst_ds.add_symlink_layer(src_ds.get_layer("color"))


@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_remote_add_symlink_mag(data_format: DataFormat) -> None:
    src_dataset_path = copy_simple_dataset(data_format, REMOTE_TESTOUTPUT_DIR)
    dst_dataset_path = prepare_dataset_path(
        data_format, REMOTE_TESTOUTPUT_DIR, "with_symlink"
    )

    src_ds = Dataset.open(src_dataset_path)
    src_layer = src_ds.get_layer("color")
    src_mag1 = src_layer.get_mag("1")

    dst_ds = Dataset(dst_dataset_path, voxel_size=(1, 1, 1))
    dst_layer = dst_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=data_format
    )
    assert not dst_layer.read_only

    with pytest.raises(AssertionError):
        dst_layer.add_symlink_mag(src_mag1)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_mag_as_copy(data_format: DataFormat, output_path: UPath) -> None:
    original_ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_ds_path = prepare_dataset_path(data_format, output_path, "copy")

    original_ds = Dataset(original_ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        data_format=data_format,
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    original_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    original_mag = original_layer.add_mag(1)
    original_mag.write(data=original_data, absolute_offset=(6, 6, 6))

    copy_ds = Dataset(copy_ds_path, voxel_size=(1, 1, 1))
    copy_layer = copy_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=data_format
    )
    copy_mag = copy_layer.add_mag_as_copy(original_mag, extend_layer_bounding_box=True)
    assert not copy_mag.read_only

    assert (copy_ds_path / "color" / "1").exists()
    assert len(copy_layer._properties.mags) == 1

    assert tuple(copy_layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(copy_layer.bounding_box.size) == (10, 20, 30)

    # Write new data in copied layer
    new_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    copy_mag.write(
        absolute_offset=(0, 0, 0),
        data=new_data,
        allow_resize=True,
        allow_unaligned=True,
    )

    np.testing.assert_array_equal(
        copy_mag.read(absolute_offset=(0, 0, 0), size=(5, 5, 5))[0], new_data
    )
    np.testing.assert_array_equal(original_mag.read()[0], original_data)

    assure_exported_properties(original_ds)
    assure_exported_properties(copy_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_fs_copy_mag(data_format: DataFormat, output_path: UPath) -> None:
    original_ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_ds_path = prepare_dataset_path(data_format, output_path, "copy")

    original_ds = Dataset(original_ds_path, voxel_size=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        data_format=data_format,
        bounding_box=BoundingBox((6, 6, 6), (10, 20, 30)),
    )
    original_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    original_mag = original_layer.add_mag(1)
    original_mag.write(data=original_data, absolute_offset=(6, 6, 6))

    copy_ds = Dataset(copy_ds_path, voxel_size=(1, 1, 1))
    copy_layer = copy_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=data_format
    )

    with mock.patch.object(
        copy_layer, "_add_fs_copy_mag", wraps=copy_layer._add_fs_copy_mag
    ) as mocked_method:
        copy_mag = copy_layer.add_mag_as_copy(
            original_mag, extend_layer_bounding_box=True
        )
        mocked_method.assert_called_once()

    assert not copy_layer.read_only
    assert not copy_mag.read_only

    assert (copy_ds_path / "color" / "1").exists()
    assert len(copy_layer._properties.mags) == 1

    assert tuple(copy_layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(copy_layer.bounding_box.size) == (10, 20, 30)

    # Write new data in copied layer
    new_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    copy_mag.write(
        absolute_offset=(0, 0, 0),
        data=new_data,
        allow_resize=True,
        allow_unaligned=True,
    )

    np.testing.assert_array_equal(
        copy_mag.read(absolute_offset=(0, 0, 0), size=(5, 5, 5))[0], new_data
    )
    np.testing.assert_array_equal(original_mag.read()[0], original_data)

    assure_exported_properties(original_ds)
    assure_exported_properties(copy_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_search_dataset_also_for_long_layer_name(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "long_layer_name")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format).add_mag("2")

    assert mag.name == "2"
    short_mag_file_path = ds.path / "color" / Mag(mag.name).to_layer_name()
    long_mag_file_path = ds.path / "color" / Mag(mag.name).to_long_layer_name()

    assert short_mag_file_path.exists()
    assert not long_mag_file_path.exists()

    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data, absolute_offset=(20, 20, 20), allow_resize=True)

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )

    # rename the path from "long_layer_name/color/2" to "long_layer_name/color/2-2-2"
    copytree(short_mag_file_path, long_mag_file_path)
    rmtree(short_mag_file_path)

    # Remove path from mag to let the path be auto-detected
    ds._properties.data_layers[0].mags[0].path = None
    ds._save_dataset_properties()

    # make sure that reading data still works
    mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20))

    # when opening the dataset, it searches both for the long and the short path
    layer = Dataset.open(ds_path).get_layer("color")
    mag = layer.get_mag("2")
    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )
    layer.delete_mag("2")

    # Note: 'ds' is outdated (it still contains Mag(2)) because it was opened again and changed.
    assure_exported_properties(layer.dataset)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_dataset_shallow_copy(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_path = prepare_dataset_path(data_format, output_path, "copy")

    ds = Dataset(ds_path, (1, 1, 1))
    ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    original_layer_1 = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel=np.uint8,
        num_channels=1,
        data_format=data_format,
    )
    original_layer_1.add_mag(1)
    original_layer_1.add_mag("2-2-1")
    original_layer_2 = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype_per_channel=np.uint32,
        largest_segment_id=0,
        data_format=data_format,
    ).as_segmentation_layer()
    original_layer_2.add_mag(4)
    agglomerates_path = original_layer_2.path / "agglomerates" / "agglomerate_view.hdf5"
    agglomerates_path.parent.mkdir(parents=True)
    agglomerates_path.touch()
    original_layer_2.attachments.add_agglomerate(
        agglomerates_path,
        name="agglomerate_view",
        data_format=AttachmentDataFormat.HDF5,
    )

    shallow_copy_of_ds = ds.shallow_copy_dataset(copy_path)
    assert (
        shallow_copy_of_ds.default_view_configuration
        and shallow_copy_of_ds.default_view_configuration.zoom == 1.5
    )
    shallow_copy_of_ds.get_layer("color").add_mag(Mag("4-4-1"))
    assert len(Dataset.open(ds_path).get_layer("color").mags) == 2, (
        "Adding a new mag should not affect the original dataset"
    )
    assert len(Dataset.open(copy_path).get_layer("color").mags) == 3, (
        "Expecting all mags from original dataset and new downsampled mag"
    )
    assert str(
        shallow_copy_of_ds.get_segmentation_layer("segmentation")
        .attachments.agglomerates[0]
        .path
    ) == str(ds_path / "segmentation" / "agglomerates" / "agglomerate_view.hdf5"), (
        "Expecting agglomerates to exist in shallow copy"
    )

    assert not (
        copy_path / "segmentation" / "agglomerates" / "agglomerate_view.hdf5"
    ).exists(), "Expecting agglomerates not to exist in shallow copy"

    assert not shallow_copy_of_ds.get_layer("color").read_only
    assert shallow_copy_of_ds.get_layer("color").get_mag(1).read_only


def test_dataset_shallow_copy_downsample() -> None:
    ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "original")
    copy_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, TESTOUTPUT_DIR, "copy")

    ds = Dataset(ds_path, (1, 1, 1))
    original_layer_1 = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_channel=np.uint8,
        num_channels=1,
        data_format=DEFAULT_DATA_FORMAT,
        bounding_box=BoundingBox((0, 0, 0), (512, 512, 512)),
    )
    original_layer_1.add_mag(1)

    # Creating a shallow copy
    shallow_copy_of_ds = ds.shallow_copy_dataset(copy_path)
    # Pre-initializing the downsampled mags
    shallow_copy_of_ds.get_layer("color").downsample(
        from_mag=Mag(1), coarsest_mag=Mag(2), only_setup_mags=True
    )
    # Re-opening the copy dataset in order to re-determine read-only mags
    shallow_copy_of_ds = Dataset.open(copy_path)
    with get_executor("sequential") as ex:
        shallow_copy_of_ds.get_layer("color").downsample(
            from_mag=Mag(1), coarsest_mag=Mag(2), allow_overwrite=True, executor=ex
        )

    assert not shallow_copy_of_ds.get_layer("color").read_only
    assert shallow_copy_of_ds.get_layer("color").get_mag(1).read_only


def test_remote_wkw_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    with pytest.raises(AssertionError):
        ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.WKW)


def test_dataset_conversion_wkw_only() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "original")
    converted_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "converted")

    # create example dataset
    origin_ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    origin_ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    seg_layer = origin_ds.add_layer(
        "layer1",
        SEGMENTATION_CATEGORY,
        num_channels=1,
        largest_segment_id=1000000000,
    )
    seg_layer.add_mag(
        "1", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8),
        allow_resize=True,
    )
    seg_layer.add_mag(
        "2", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8),
        allow_resize=True,
    )
    wk_color_layer = origin_ds.add_layer("layer2", COLOR_CATEGORY, num_channels=3)
    wk_color_layer.add_mag(
        "1", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
        allow_resize=True,
    )
    wk_color_layer.add_mag(
        "2", chunk_shape=Vec3Int.full(8), shard_shape=Vec3Int.full(128)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8),
        allow_resize=True,
    )
    converted_ds = origin_ds.copy_dataset(converted_path)

    assert (
        converted_ds.default_view_configuration
        and converted_ds.default_view_configuration.zoom == 1.5
    )
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
            assert origin_info.chunk_shape == converted_info.chunk_shape
            assert origin_info.data_format == converted_info.data_format
            np.testing.assert_array_equal(
                origin_ds.layers[layer_name].mags[mag].read(),
                converted_ds.layers[layer_name].mags[mag].read(),
            )

    assure_exported_properties(origin_ds)
    assure_exported_properties(converted_ds)


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_dataset_conversion_from_wkw_to_zarr(
    output_path: UPath, data_format: DataFormat
) -> None:
    converted_path = prepare_dataset_path(data_format, output_path, "converted")

    input_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")
    print(input_ds.get_layer("color").get_mag("1").info.chunk_shape)
    converted_ds = input_ds.copy_dataset(
        converted_path,
        data_format=data_format,
        shard_shape=8 if data_format == DataFormat.Zarr else 32,
    )

    if data_format == DataFormat.Zarr:
        assert (converted_path / "color" / "1" / ".zarray").exists()
    else:
        assert (converted_path / "color" / "1" / "zarr.json").exists()
    assert np.all(
        input_ds.get_layer("color").get_mag("1").read()
        == converted_ds.get_layer("color").get_mag("1").read()
    )
    assert converted_ds.get_layer("color").data_format == data_format
    assert converted_ds.get_layer("color").get_mag("1").info.data_format == data_format

    assure_exported_properties(converted_ds)


@pytest.mark.parametrize(
    "data_format", DATA_FORMATS
)  # Don't test remote storage for performance reasons (lack of sharding in zarr)
def test_for_zipped_chunks(data_format: DataFormat) -> None:
    src_dataset_path = prepare_dataset_path(
        data_format, TESTOUTPUT_DIR, "zipped_chunking_source"
    )
    dst_dataset_path = prepare_dataset_path(
        data_format, TESTOUTPUT_DIR, "zipped_chunking_target"
    )

    ds = Dataset(src_dataset_path, voxel_size=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag("1")
    mag.write(
        data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8),
        allow_resize=True,
    )
    source_view = mag.get_view(absolute_offset=(0, 0, 0), size=(256, 256, 256))

    target_mag = (
        Dataset(dst_dataset_path, voxel_size=(1, 1, 2))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint8",
            num_channels=3,
            data_format=data_format,
        )
        .get_or_add_mag(
            "1",
            chunk_shape=Vec3Int.full(8),
            shard_shape=(32 if data_format != DataFormat.Zarr else 8),
        )
    )

    target_mag.layer.bounding_box = BoundingBox((0, 0, 0), (256, 256, 256))
    target_view = target_mag.get_view(absolute_offset=(0, 0, 0), size=(256, 256, 256))

    with get_executor("sequential") as executor:
        func = named_partial(
            copy_and_transform_job, name="foo", val=42
        )  # curry the function with further arguments
        source_view.for_zipped_chunks(
            func,
            target_view=target_view,
            source_chunk_shape=(64, 64, 64),  # multiple of (wkw_file_len,) * 3
            target_chunk_shape=(64, 64, 64),  # multiple of (wkw_file_len,) * 3
            executor=executor,
        )

    np.testing.assert_array_equal(
        source_view.read() + 50,
        target_view.read(),
    )

    assure_exported_properties(ds)


def _func_invalid_target_chunk_shape_wk(args: tuple[View, View, int]) -> None:
    (_s, _t, _i) = args


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_for_zipped_chunks_invalid_target_chunk_shape_wk(
    data_format: DataFormat, output_path: UPath
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "zipped_chunking_source_invalid"
    )
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer1 = ds.get_or_add_layer("color1", COLOR_CATEGORY, data_format=data_format)
    source_mag_view = layer1.get_or_add_mag(
        1, chunk_shape=chunk_shape, shard_shape=shard_shape
    )

    layer2 = ds.get_or_add_layer("color2", COLOR_CATEGORY, data_format=data_format)
    target_mag_view = layer2.get_or_add_mag(
        1, chunk_shape=chunk_shape, shard_shape=shard_shape
    )

    source_view = source_mag_view.get_view(
        absolute_offset=(0, 0, 0), size=(300, 300, 300), read_only=True
    )
    layer2.bounding_box = BoundingBox((0, 0, 0), (300, 300, 300))
    target_view = target_mag_view.get_view()

    with get_executor("sequential") as executor:
        for test_case in test_cases_wk:
            with pytest.raises(AssertionError):
                source_view.for_zipped_chunks(
                    func_per_chunk=_func_invalid_target_chunk_shape_wk,
                    target_view=target_view,
                    source_chunk_shape=test_case,
                    target_chunk_shape=test_case,
                    executor=executor,
                )

    assure_exported_properties(ds)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_invalid_chunk_shard_shape(output_path: UPath) -> None:
    ds_path = prepare_dataset_path(
        DEFAULT_DATA_FORMAT, output_path, "invalid_chunk_shape"
    )
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DEFAULT_DATA_FORMAT)

    with pytest.raises(ValueError, match=".*must be a power of two.*"):
        layer.add_mag("1", chunk_shape=(3, 4, 4))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(8, 16, 16))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(8, 8, 8))

    with pytest.raises(ValueError, match=".*must be a multiple.*"):
        # also not a power-of-two shard shape
        layer.add_mag("1", chunk_shape=(16, 16, 16), shard_shape=(53, 16, 16))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_only_view(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "read_only_view")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag = ds.get_or_add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    ).get_or_add_mag("1")
    mag.write(
        data=(np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8),
        absolute_offset=(10, 20, 30),
        allow_resize=True,
        allow_unaligned=True,
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = (np.random.rand(1, 5, 6, 7) * 255).astype(np.uint8)
    with pytest.raises(RuntimeError):
        v_read.write(data=new_data)

    v_write.write(data=new_data, allow_unaligned=True)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_bounding_box_on_disk(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    chunk_shape, shard_shape = default_chunk_config(data_format, 8)
    mag = ds.add_layer("color", category="color", data_format=data_format).add_mag(
        "2-2-1", chunk_shape=chunk_shape, shard_shape=shard_shape
    )  # cube_size = 8*8 = 64

    write_positions = [
        Vec3Int(0, 0, 0),
        Vec3Int(20, 80, 120),
        Vec3Int(1000, 2000, 4000),
    ]
    data_size = Vec3Int(10, 20, 30)
    write_data = (np.random.rand(*data_size) * 255).astype(np.uint8)
    for offset in write_positions:
        mag.write(
            absolute_offset=offset * mag.mag.to_vec3_int(),
            data=write_data,
            allow_resize=True,
            allow_unaligned=True,
        )

    if is_remote_path(output_path):
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            bounding_boxes_on_disk = list(mag.get_bounding_boxes_on_disk())

        assert (
            len(bounding_boxes_on_disk)
            == mag.bounding_box.size.ceildiv(mag._array.info.shard_shape)
            .ceildiv(mag.mag)
            .prod()
        )
    else:
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_compression(data_format: DataFormat, output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag(1, compress=False)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert not mag1._is_compressed()

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Remote datasets require a `target_path` for compression
        with pytest.raises(AssertionError):
            mag1.compress()

        compressed_dataset_path = (
            REMOTE_TESTOUTPUT_DIR / f"simple_{data_format}_dataset_compressed"
        )
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            mag1.compress(
                target_path=compressed_dataset_path,
            )
        mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    else:
        with get_executor("sequential") as executor:
            mag1.compress(executor=executor)

    assert mag1._is_compressed()
    assert mag1.info.data_format == data_format

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    # writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
    mag1.write(
        (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8), allow_resize=True
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_rechunking(data_format: DataFormat, output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag(
        1,
        compress=False,
        chunk_shape=(16, 16, 16),
        shard_shape=(16, 16, 16) if data_format == DataFormat.Zarr else (64, 64, 64),
    )

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert not mag1._is_compressed()

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Remote datasets require a `target_path` for rechunking
        with pytest.raises(AssertionError):
            mag1.rechunk()

        compressed_dataset_path = (
            REMOTE_TESTOUTPUT_DIR / f"simple_{data_format}_dataset_compressed"
        )
        with pytest.warns(UserWarning, match=".*can be slow.*"):
            mag1.rechunk(
                target_path=compressed_dataset_path,
            )
        mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    else:
        with get_executor("sequential") as executor:
            mag1.rechunk(executor=executor)

    assert mag1.info.data_format == data_format
    assert mag1._is_compressed()
    assert mag1.info.chunk_shape == Vec3Int.full(32)
    if data_format == DataFormat.Zarr:
        assert mag1.info.shard_shape == Vec3Int.full(32)
    else:
        assert mag1.info.shard_shape == Vec3Int.full(1024)

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    # writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
    mag1.write(
        (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8), allow_resize=True
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_config(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(
        1,
        compress=Zarr3Config(
            codecs=(
                {"name": "bytes"},
                {"name": "gzip", "configuration": {"level": 3}},
            ),
            chunk_key_encoding={
                "name": "default",
                "configuration": {"separator": "."},
            },
        ),
    )

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert isinstance(mag1.info, Zarr3ArrayInfo)
    assert mag1.info.codecs == (
        {"name": "bytes"},
        {"name": "gzip", "configuration": {"level": 3}},
    )
    assert mag1.info.chunk_key_encoding == {
        "name": "default",
        "configuration": {"separator": "."},
    }
    assert (mag1.path / "c.0.0.0.0").exists()
    assert json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0][
        "configuration"
    ]["codecs"] == [
        {"name": "bytes"},
        {"name": "gzip", "configuration": {"level": 3}},
    ]

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_sharding(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(1, chunk_shape=(32, 32, 32), shard_shape=(64, 64, 64))

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    assert (
        json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0]["name"]
        == "sharding_indexed"
    )

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_zarr3_no_sharding(output_path: UPath) -> None:
    new_dataset_path = prepare_dataset_path(DataFormat.Zarr3, output_path)
    ds = Dataset(new_dataset_path, voxel_size=(2, 2, 1))
    mag1 = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr3
    ).add_mag(1, chunk_shape=(32, 32, 32), shard_shape=(32, 32, 32))

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100), allow_resize=True)

    # Don't set up a sharding codec, if no sharding is necessary, i.e. chunk_shape == shard_shape
    assert (
        json.loads((mag1.path / "zarr.json").read_bytes())["codecs"][0]["name"]
        != "sharding_indexed"
    )

    np.testing.assert_array_equal(
        write_data, mag1.read(absolute_offset=(60, 80, 100), size=(10, 20, 30))
    )

    assure_exported_properties(mag1.layer.dataset)


def test_dataset_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, voxel_size=(2, 2, 1))
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is None

    ds1.default_view_configuration = DatasetViewConfiguration(four_bit=True)
    default_view_configuration = ds1.default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.four_bit == True
    assert default_view_configuration.interpolation is None
    assert default_view_configuration.render_missing_data_black is None
    assert default_view_configuration.loading_strategy is None
    assert default_view_configuration.segmentation_pattern_opacity is None
    assert default_view_configuration.zoom is None
    assert default_view_configuration.position is None
    assert default_view_configuration.rotation is None

    # Test if only the set parameters are stored in the properties
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
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
    ds2 = Dataset.open(ds_path)
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
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
    view_configuration_dict = properties["defaultViewConfiguration"]
    for k in view_configuration_dict.keys():
        assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_layer_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, voxel_size=(2, 2, 1))
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
    properties = json.loads((ds1.path / PROPERTIES_FILE_NAME).read_text())
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
    ds2 = Dataset.open(ds_path)
    default_view_configuration = ds2.get_layer("color").default_view_configuration
    assert default_view_configuration is not None
    assert default_view_configuration.color == (255, 0, 0)
    assert default_view_configuration.alpha == 1.0
    assert default_view_configuration.intensity_range == (-12.3e1, 123)
    assert default_view_configuration.is_inverted == True
    assert default_view_configuration.min == 55.0

    # Test camel case
    properties = json.loads((ds2.path / PROPERTIES_FILE_NAME).read_text())
    view_configuration_dict = properties["dataLayers"][0]["defaultViewConfiguration"]
    for k in view_configuration_dict.keys():
        assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_get_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    segmentation_layer = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    ).as_segmentation_layer()
    assert segmentation_layer.largest_segment_id == 999
    segmentation_layer.largest_segment_id = 123
    assert segmentation_layer.largest_segment_id == 123

    assure_exported_properties(ds)


def test_refresh_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    segmentation_layer = ds.add_layer(
        "segmentation", SEGMENTATION_CATEGORY
    ).as_segmentation_layer()
    mag = segmentation_layer.add_mag(Mag(1))

    assert segmentation_layer.largest_segment_id is None

    write_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    mag.write(data=write_data, allow_resize=True)

    segmentation_layer.refresh_largest_segment_id()

    assert segmentation_layer.largest_segment_id == np.max(write_data, initial=0)


def test_get_or_add_layer_by_type() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
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


def test_read_bbox() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag(1)
    mag.write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(50, 60, 70) * 255).astype(np.uint8),
        allow_resize=True,
    )

    np.testing.assert_array_equal(
        mag.read(absolute_offset=(20, 30, 40), size=(40, 50, 60)),
        mag.read(
            absolute_bounding_box=BoundingBox(topleft=(20, 30, 40), size=(40, 50, 60))
        ),
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_as_copy(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_path = prepare_dataset_path(data_format, output_path, "copy")

    ds = Dataset(ds_path, voxel_size=(2, 2, 1))

    # Create dataset to copy data from
    other_ds = Dataset(copy_path, voxel_size=(2, 2, 1))
    original_color_layer = other_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    )
    original_color_layer.add_mag(1).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(32, 64, 128) * 255).astype(np.uint8),
        allow_resize=True,
    )
    other_ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        data_format=data_format,
        largest_segment_id=999,
    ).add_mag("1")

    # Copies the "color" layer from a different dataset
    ds.add_layer_as_copy(copy_path / "color")
    ds.add_layer_as_copy(copy_path / "segmentation")
    assert len(ds.layers) == 2
    assert (
        ds.get_layer("segmentation").as_segmentation_layer().largest_segment_id == 999
    )

    color_layer = ds.get_layer("color")
    assert color_layer.bounding_box == BoundingBox(
        topleft=(10, 20, 30), size=(32, 64, 128)
    )
    assert color_layer.mags.keys() == original_color_layer.mags.keys()
    assert len(color_layer.mags.keys()) >= 1
    for mag in color_layer.mags.keys():
        np.testing.assert_array_equal(
            color_layer.get_mag(mag).read(), original_color_layer.get_mag(mag).read()
        )
        # Test if the copied layer contains actual data
        assert np.max(color_layer.get_mag(mag).read()) > 0

    with pytest.raises(IndexError):
        # The dataset already has a layer called "color".
        ds.add_layer_as_copy(copy_path / "color")

    # Test if the changes of the properties are persisted on disk by opening it again
    assert "color" in Dataset.open(ds_path).layers.keys()

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_rename_layer(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(1)
    write_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    mag.write(data=write_data, allow_resize=True)

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Cannot rename layers on remote storage
        with pytest.raises(RuntimeError):
            layer.name = "color2"
        return
    else:
        layer.name = "color2"

    assert not (ds_path / "color").exists()
    assert (ds_path / "color2").exists()
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 0
    )
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color2"])
        == 1
    )
    assert ds._properties.data_layers[0].mags[0].path == "./color2/1"
    assert "color2" in ds.layers.keys()
    assert "color" not in ds.layers.keys()
    assert ds.get_layer("color2").data_format == data_format

    # The "mag" object which was created before renaming the layer is still valid
    np.testing.assert_array_equal(mag.read()[0], write_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_delete_layer_and_mag(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    color_layer.add_mag(1)
    color_layer.add_mag(2)
    ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
        data_format=data_format,
    )
    assert "color" in ds.layers
    assert "segmentation" in ds.layers
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 1
    )
    assert (
        len(
            [
                layer
                for layer in ds._properties.data_layers
                if layer.name == "segmentation"
            ]
        )
        == 1
    )
    assert len(color_layer._properties.mags) == 2

    color_layer.delete_mag(1)
    assert len(color_layer._properties.mags) == 1
    assert len([m for m in color_layer._properties.mags if Mag(m.mag) == Mag(2)]) == 1

    ds.delete_layer("color")
    assert "color" not in ds.layers
    assert "segmentation" in ds.layers
    assert (
        len([layer for layer in ds._properties.data_layers if layer.name == "color"])
        == 0
    )
    assert (
        len(
            [
                layer
                for layer in ds._properties.data_layers
                if layer.name == "segmentation"
            ]
        )
        == 1
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_like(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    color_layer1 = ds.add_layer(
        "color1",
        COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=data_format,
    )
    color_layer1.add_mag(1)
    segmentation_layer1 = ds.add_layer(
        "segmentation1",
        SEGMENTATION_CATEGORY,
        dtype_per_channel="uint8",
        largest_segment_id=999,
        data_format=data_format,
    ).as_segmentation_layer()
    segmentation_layer1.add_mag(1)
    color_layer2 = ds.add_layer_like(color_layer1, "color2")
    segmentation_layer2 = ds.add_layer_like(
        segmentation_layer1, "segmentation2"
    ).as_segmentation_layer()

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
    assert color_layer1.data_format == color_layer2.data_format == data_format

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
        segmentation_layer1.data_format
        == segmentation_layer2.data_format
        == data_format
    )
    assert (
        segmentation_layer1.largest_segment_id
        == segmentation_layer2.largest_segment_id
        == 999
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize(
    "dtype_per_channel,category,is_supported",
    [
        ("uint8", COLOR_CATEGORY, True),
        ("uint16", COLOR_CATEGORY, True),
        ("uint32", COLOR_CATEGORY, True),
        ("uint64", COLOR_CATEGORY, False),
        ("int8", COLOR_CATEGORY, True),
        ("int16", COLOR_CATEGORY, True),
        ("int32", COLOR_CATEGORY, True),
        ("int64", COLOR_CATEGORY, False),
        ("float32", COLOR_CATEGORY, True),
        ("float64", COLOR_CATEGORY, False),
        ("uint8", SEGMENTATION_CATEGORY, True),
        ("uint16", SEGMENTATION_CATEGORY, True),
        ("uint32", SEGMENTATION_CATEGORY, True),
        ("uint64", SEGMENTATION_CATEGORY, True),
        ("int8", SEGMENTATION_CATEGORY, True),
        ("int16", SEGMENTATION_CATEGORY, True),
        ("int32", SEGMENTATION_CATEGORY, True),
        ("int64", SEGMENTATION_CATEGORY, True),
        ("float32", SEGMENTATION_CATEGORY, False),
        ("float64", SEGMENTATION_CATEGORY, False),
    ],
)
def test_add_layer_dtype_per_channel(
    dtype_per_channel: str, category: LayerCategoryType, is_supported: bool
) -> None:
    ds_path = prepare_dataset_path(
        DataFormat.Zarr3, TESTOUTPUT_DIR, "dtype_per_channel"
    )
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    if is_supported:
        layer = ds.add_layer(
            "test_layer",
            category=category,
            dtype_per_channel=dtype_per_channel,
        )
        assert layer.dtype_per_channel == np.dtype(dtype_per_channel)
    else:
        with pytest.raises(
            ValueError,
            match="Supported dtypes are:",
        ):
            ds.add_layer(
                "test_layer",
                category=category,
                dtype_per_channel=dtype_per_channel,
            )


def test_pickle_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "pickle")
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    mag1 = ds.add_layer("color", COLOR_CATEGORY).add_mag(1)

    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag1.write(data_to_write, allow_resize=True)
    assert mag1._cached_array is not None

    with (ds_path / "save.p").open("wb") as f_write:
        pickle.dump(mag1, f_write)
    with (ds_path / "save.p").open("rb") as f_read:
        pickled_mag1 = pickle.load(f_read)

    # Make sure that the pickled mag can still read data
    assert pickled_mag1._cached_array is None
    np.testing.assert_array_equal(
        data_to_write,
        pickled_mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )
    assert pickled_mag1._cached_array is not None

    # Make sure that the attributes of the MagView (not View) still exist
    assert pickled_mag1.layer is not None


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_warn_outdated_properties(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds1 = Dataset(ds_path, voxel_size=(1, 1, 1))
    ds2 = Dataset.open(ds_path)

    # Change ds1 and undo it again
    ds1.add_layer("color", COLOR_CATEGORY, data_format=data_format).add_mag(1)
    ds1.delete_layer("color")

    # Changing ds2 should work fine, since the properties on disk
    # haven't changed.
    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        data_format=data_format,
        largest_segment_id=1,
    ).add_mag(1)

    with pytest.raises(UserWarning):
        # Changing ds1 should raise a warning, since ds1
        # does not know about the change in ds2
        ds1.add_layer("color", COLOR_CATEGORY, data_format=data_format)


def test_dataset_properties_version() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    properties_path = ds.path / PROPERTIES_FILE_NAME
    properties = json.loads((properties_path).read_bytes())
    assert properties["version"] == 1

    # write invalid version
    properties["version"] = 9000
    properties_path.write_text(json.dumps(properties))

    with pytest.raises(AssertionError):
        Dataset.open(ds_path)


def test_can_compress_mag8() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))

    layer = ds.add_layer("color", COLOR_CATEGORY)
    layer.bounding_box = BoundingBox((0, 0, 0), (12240, 12240, 685))
    for mag in ["1", "2-2-1", "4-4-1", "8-8-2"]:
        layer.add_mag(mag, compress=False)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (12240, 12240, 685))

    mag_view = layer.get_mag("8-8-2")
    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag_view.write(
        data_to_write, absolute_offset=(11264, 11264, 0), allow_unaligned=True
    )
    mag_view.compress()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_downsampling(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "downsampling")

    color_layer = Dataset.open(ds_path).get_layer("color")
    color_layer.downsample()

    assert (ds_path / "color" / "2").exists()
    assert (ds_path / "color" / "4").exists()

    if data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "2" / ".zarray").exists()
        assert (ds_path / "color" / "4" / ".zarray").exists()
    elif data_format == DataFormat.Zarr3:
        assert (ds_path / "color" / "2" / "zarr.json").exists()
        assert (ds_path / "color" / "4" / "zarr.json").exists()
    else:
        assert (ds_path / "color" / "2" / "header.wkw").exists()
        assert (ds_path / "color" / "4" / "header.wkw").exists()

    assure_exported_properties(color_layer.dataset)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_aligned_downsampling(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "aligned_downsampling")
    dataset = Dataset.open(ds_path)
    input_layer = dataset.get_layer("color")
    input_layer.downsample(coarsest_mag=Mag(2))
    test_layer = dataset.add_layer(
        layer_name="color_2",
        category="color",
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=input_layer.data_format,
    )

    shard_shape = None
    if data_format == DataFormat.Zarr3:
        # Writing compressed zarr with large shard shape is slow
        # compare https://github.com/scalableminds/webknossos-libs/issues/964
        shard_shape = (128, 128, 128)

    test_mag = test_layer.add_mag("1", shard_shape=shard_shape)
    test_mag.write(
        absolute_offset=(0, 0, 0),
        # assuming the layer has 3 channels:
        data=(np.random.rand(3, 24, 24, 24) * 255).astype(np.uint8),
        allow_resize=True,
    )
    test_layer.downsample(coarsest_mag=Mag(2))

    assert (ds_path / "color_2" / "1").exists()
    assert (ds_path / "color_2" / "2").exists()

    if data_format == DataFormat.Zarr:
        assert (ds_path / "color_2" / "1" / ".zarray").exists()
        assert (ds_path / "color_2" / "2" / ".zarray").exists()
    elif data_format == DataFormat.Zarr3:
        assert (ds_path / "color_2" / "1" / "zarr.json").exists()
        assert (ds_path / "color_2" / "2" / "zarr.json").exists()
    else:
        assert (ds_path / "color_2" / "1" / "header.wkw").exists()
        assert (ds_path / "color_2" / "2" / "header.wkw").exists()

    assure_exported_properties(dataset)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_guided_downsampling(data_format: DataFormat, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "guided_downsampling")

    input_dataset = Dataset.open(ds_path)
    input_layer = input_dataset.get_layer("color")

    shard_shape = None
    if data_format == DataFormat.Zarr3:
        # Writing compressed zarr with large shard shape is slow
        # compare https://github.com/scalableminds/webknossos-libs/issues/964
        shard_shape = (128, 128, 128)

    # Adding additional mags to the input dataset for testing
    input_layer.add_mag("2-2-1", shard_shape=shard_shape)
    input_layer.redownsample()
    assert len(input_layer.mags) == 2
    # Use the mag with the best resolution
    finest_input_mag = input_layer.get_finest_mag()

    # Creating an empty dataset for testing
    output_ds_path = ds_path.parent / (ds_path.name + "_output")
    output_dataset = Dataset(output_ds_path, voxel_size=input_dataset.voxel_size)
    output_layer = output_dataset.add_layer(
        layer_name="color",
        category="color",
        dtype_per_channel=input_layer.dtype_per_channel,
        num_channels=input_layer.num_channels,
        data_format=input_layer.data_format,
    )
    # Create the same mag in the new output dataset
    output_mag = output_layer.add_mag(finest_input_mag.mag, shard_shape=shard_shape)
    # Copying some data into the output dataset
    input_data = finest_input_mag.read(absolute_offset=(0, 0, 0), size=(24, 24, 24))
    output_mag.write(absolute_offset=(0, 0, 0), data=input_data, allow_resize=True)
    # Downsampling the layer to the magnification used in the input dataset
    output_layer.downsample(
        from_mag=output_mag.mag,
        coarsest_mag=Mag("4-4-2"),
        align_with_other_layers=input_dataset,
    )
    for mag in input_layer.mags:
        assert output_layer.get_mag(mag)

    assert (output_ds_path / "color" / "1").exists()
    assert (output_ds_path / "color" / "2-2-1").exists()
    assert (output_ds_path / "color" / "4-4-2").exists()

    if data_format == DataFormat.Zarr:
        assert (output_ds_path / "color" / "1" / ".zarray").exists()
        assert (output_ds_path / "color" / "2-2-1" / ".zarray").exists()
        assert (output_ds_path / "color" / "4-4-2" / ".zarray").exists()
    elif data_format == DataFormat.Zarr3:
        assert (output_ds_path / "color" / "1" / "zarr.json").exists()
        assert (output_ds_path / "color" / "2-2-1" / "zarr.json").exists()
        assert (output_ds_path / "color" / "4-4-2" / "zarr.json").exists()
    else:
        assert (output_ds_path / "color" / "1" / "header.wkw").exists()
        assert (output_ds_path / "color" / "2-2-1" / "header.wkw").exists()
        assert (output_ds_path / "color" / "4-4-2" / "header.wkw").exists()

    assure_exported_properties(input_dataset)


@pytest.mark.parametrize("data_format", [DataFormat.Zarr, DataFormat.Zarr3])
def test_zarr_copy_to_remote_dataset(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, REMOTE_TESTOUTPUT_DIR, "copied")
    Dataset.open(TESTDATA_DIR / "simple_zarr_dataset").copy_dataset(
        ds_path,
        shard_shape=32,
        data_format=data_format,
    )
    if data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "1" / ".zarray").exists()
    else:
        assert (ds_path / "color" / "1" / "zarr.json").exists()


@pytest.mark.parametrize("input_path", OUTPUT_PATHS)
@pytest.mark.parametrize("output_path", OUTPUT_PATHS)
def test_copy_dataset_with_attachments(input_path: UPath, output_path: UPath) -> None:
    ds_path = copy_simple_dataset(DEFAULT_DATA_FORMAT, input_path)
    new_ds_path = prepare_dataset_path(DEFAULT_DATA_FORMAT, output_path, "copied")

    ds = Dataset.open(ds_path)
    ds.default_view_configuration = DatasetViewConfiguration(zoom=1.5)
    # Add segmentation layer and meshfile
    seg_layer = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=999,
        bounding_box=BoundingBox((0, 0, 0), (10, 10, 10)),
    ).as_segmentation_layer()
    seg_mag = seg_layer.add_mag(1)
    seg_mag.write(data=np.zeros((10, 10, 10), dtype=np.uint8))

    meshfile_path = seg_layer.path / "meshes" / "meshfile"
    meshfile_path.mkdir(parents=True, exist_ok=True)
    (meshfile_path / "zarr.json").write_text("test")

    seg_layer.attachments.add_mesh(
        meshfile_path,
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    # Copy
    copy_ds = ds.copy_dataset(new_ds_path)

    assert (
        copy_ds.default_view_configuration
        and copy_ds.default_view_configuration.zoom == 1.5
    )
    assert (new_ds_path / "segmentation" / "1" / "zarr.json").exists()
    assert (new_ds_path / "segmentation" / "meshes" / "meshfile" / "zarr.json").exists()


def test_wkw_copy_to_remote_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied")
    wkw_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")

    # Fails with explicit data_format=wkw ...
    with pytest.raises(AssertionError):
        wkw_ds.copy_dataset(ds_path, shard_shape=32, data_format=DataFormat.WKW)

    # ... and with implicit data_format=wkw from the source layers.
    with pytest.raises(AssertionError):
        wkw_ds.copy_dataset(
            ds_path,
            shard_shape=32,
        )


def test_copy_dataset_exists_ok() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied")
    wkw_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")

    wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3)
    with pytest.raises(RuntimeError):
        wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3)
    wkw_ds.copy_dataset(ds_path, data_format=DataFormat.Zarr3, exists_ok=True)


@pytest.mark.use_proxay
def test_remote_dataset_access_metadata() -> None:
    ds = RemoteDataset.open("l4_sample", "Organization_X")
    assert len(ds.metadata) == 0

    ds.metadata["key"] = "value"
    assert ds.metadata["key"] == "value"

    ds.metadata["number"] = 42
    assert ds.metadata["number"] == 42

    ds.metadata["list"] = ["a", "b", "c"]
    assert ds.metadata["list"] == ["a", "b", "c"]

    assert len(ds.folder.metadata) == 1

    ds.folder.metadata["folder_key"] = "folder_value"
    assert ds.folder.metadata["folder_key"] == "folder_value"
    assert len(ds.folder.metadata) == 2


@pytest.mark.use_proxay
def test_remote_dataset_urls() -> None:
    ds = RemoteDataset.open("l4_sample", "Organization_X")
    dataset_id = ds._dataset_id
    assert dataset_id in ds.url

    ds_open_with_id = RemoteDataset.open(dataset_id=dataset_id)
    assert ds_open_with_id.url == ds.url

    # Test different variants of the URL
    # 1. deprecated url: "http://localhost:9000/datasets/Organization_X/l4_sample"

    ds1 = RemoteDataset.open("http://localhost:9000/datasets/Organization_X/l4_sample")
    assert ds1.url == ds.url

    # 2. deprecated url with params: "http://localhost:9000/datasets/Organization_X/l4_sample/view#2786,4326,1816,0,3"
    ds2 = RemoteDataset.open(
        "http://localhost:9000/datasets/Organization_X/l4_sample/view#2786,4326,1816,0,3"
    )
    assert ds2.url == ds.url

    # 3. new url: "http://localhost:9000/datasets/{dataset_id}"
    ds3 = RemoteDataset.open(f"http://localhost:9000/datasets/{dataset_id}")
    assert ds3.url == ds.url

    # 4. new url with params: "http://localhost:9000/datasets/{dataset_id}/view#2786,4326,1816,0,3"
    ds4 = RemoteDataset.open(
        f"http://localhost:9000/datasets/{dataset_id}/view#2786,4326,1816,0,3"
    )
    assert ds4.url == ds.url

    # 5. new url with ds name: "http://localhost:9000/datasets/l4_sample-{dataset_id}"
    ds5 = RemoteDataset.open(f"http://localhost:9000/datasets/l4_sample-{dataset_id}")
    assert ds5.url == ds.url

    # 6. new url with ds name and params: "http://localhost:9000/datasets/l4_sample-{dataset_id}/view#2786,4326,1816,0,3"
    ds6 = RemoteDataset.open(
        f"http://localhost:9000/datasets/l4_sample-{dataset_id}/view#2786,4326,1816,0,3"
    )
    assert ds6.url == ds.url


def test_dataset_open_wrong_path() -> None:
    with pytest.raises(FileNotFoundError):
        Dataset.open("wrong_path")


@pytest.mark.parametrize(
    "data_format", [DataFormat.N5, DataFormat.NeuroglancerPrecomputed]
)
def test_n5_and_ng_datasets(data_format: DataFormat) -> None:
    reference_data = (
        Dataset.open(TESTDATA_DIR / "simple_zarr3_dataset")
        .get_layer("color")
        .get_mag(1)
        .read()
    )

    short_data_format = "n5" if data_format == DataFormat.N5 else "ng"

    test_mag = (
        Dataset.open(TESTDATA_DIR / f"simple_{short_data_format}_dataset")
        .get_layer("color")
        .get_mag(1)
    )
    assert test_mag.layer.data_format == data_format

    test_data = test_mag.read()
    np.testing.assert_equal(test_data, reference_data)

    with pytest.raises(RuntimeError):
        test_mag.write(
            absolute_offset=(0, 0, 0), data=np.ones((3, 24, 24, 24), dtype="uint8")
        )

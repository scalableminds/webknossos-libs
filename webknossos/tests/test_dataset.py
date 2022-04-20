import itertools
import json
import pickle
import shlex
import subprocess
import warnings
from pathlib import Path
from typing import Iterator, Optional, Tuple, cast

import numpy as np
import pytest
from upath import UPath

from webknossos.dataset import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    Dataset,
    SegmentationLayer,
    View,
)
from webknossos.dataset._array import DataFormat
from webknossos.dataset.dataset import PROPERTIES_FILE_NAME
from webknossos.dataset.properties import (
    DatasetProperties,
    DatasetViewConfiguration,
    LayerViewConfiguration,
    SegmentationLayerProperties,
    dataset_converter,
)
from webknossos.geometry import BoundingBox, Mag, Vec3Int
from webknossos.utils import (
    copytree,
    get_executor_for_args,
    named_partial,
    rmtree,
    snake_to_camel_case,
)

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR

MINIO_ROOT_USER = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
MINIO_ROOT_PASSWORD = "ANTN35UAENTS5UIAEATD"
MINIO_PORT = "8000"


@pytest.fixture(autouse=True, scope="module")
def docker_minio() -> Iterator[None]:
    """Minio is an S3 clone and is used as local test server"""
    container_name = "minio"
    cmd = (
        "docker run"
        f" -p {MINIO_PORT}:9000"
        f" -e MINIO_ROOT_USER={MINIO_ROOT_USER}"
        f" -e MINIO_ROOT_PASSWORD={MINIO_ROOT_PASSWORD}"
        f" --name {container_name}"
        " --rm"
        " -d"
        " minio/minio server /data"
    )
    print("BEFORE", flush=True)
    subprocess.check_output(shlex.split(cmd))
    REMOTE_TESTOUTPUT_DIR.fs.mkdirs("testoutput", exist_ok=True)
    try:
        yield
    finally:
        subprocess.check_output(["docker", "stop", container_name])


REMOTE_TESTOUTPUT_DIR = UPath(
    "s3://testoutput",
    key=MINIO_ROOT_USER,
    secret=MINIO_ROOT_PASSWORD,
    client_kwargs={"endpoint_url": f"http://localhost:{MINIO_PORT}"},
)

DATA_FORMATS = [DataFormat.WKW, DataFormat.Zarr]
DATA_FORMATS_AND_OUTPUT_PATHS = [
    (DataFormat.WKW, TESTOUTPUT_DIR),
    (DataFormat.Zarr, TESTOUTPUT_DIR),
    (DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR),
]

pytestmark = [pytest.mark.block_network(allowed_hosts=[".*"])]


def copy_simple_dataset(
    data_format: DataFormat, output_path: Path, suffix: Optional[str] = None
) -> Path:
    suffix = (f"_{suffix}") if suffix is not None else ""
    new_dataset_path = output_path / f"simple_{data_format}_dataset{suffix}"
    rmtree(new_dataset_path)
    copytree(
        TESTDATA_DIR / f"simple_{data_format}_dataset",
        new_dataset_path,
    )
    return new_dataset_path


def prepare_dataset_path(
    data_format: DataFormat, output_path: Path, suffix: Optional[str] = None
) -> Path:
    suffix = (f"_{suffix}") if suffix is not None else ""
    new_dataset_path = output_path / f"{data_format}_dataset{suffix}"
    rmtree(new_dataset_path)
    return new_dataset_path


def chunk_job(args: Tuple[View, int]) -> None:
    (view, _i) = args
    # increment the color value of each voxel
    data = view.read()
    if data.shape[0] == 1:
        data = data[0, :, :, :]
    data += 50
    view.write(data)


def default_chunk_config(
    data_format: DataFormat, chunk_size: int = 32
) -> Tuple[Vec3Int, Vec3Int]:
    if data_format == DataFormat.Zarr:
        return (Vec3Int.full(chunk_size * 8), Vec3Int.full(1))
    else:
        return (Vec3Int.full(chunk_size), Vec3Int.full(8))


def advanced_chunk_job(args: Tuple[View, int], dtype: type) -> None:
    view, _i = args

    # write different data for each chunk (depending on the topleft of the chunk)
    data = view.read()
    data = np.ones(data.shape, dtype=dtype) * dtype(sum(view.bounding_box.topleft))
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
        chunk = (
            ds.get_layer("color")
            .get_mag("1")
            .get_view(absolute_offset=offset, size=size)
        )
        chunk_data = chunk.read()
        assert np.array_equal(
            np.ones(chunk_data.shape, dtype=np.uint8)
            * np.uint8(sum(chunk.bounding_box.topleft)),
            chunk_data,
        )


def copy_and_transform_job(args: Tuple[View, View, int], name: str, val: int) -> None:
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_create_dataset_with_layer_and_mag(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)

    ds = Dataset(ds_path, scale=(1, 1, 1))
    ds.add_layer("color", "color", data_format=data_format)

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    if data_format == DataFormat.WKW:
        assert (ds_path / "color" / "1" / "header.wkw").exists()
        assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()
    elif data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "1" / ".zarray").exists()
        assert (ds_path / "color" / "2-2-1" / ".zarray").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assure_exported_properties(ds)


def test_create_default_layer() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)

    assert layer.data_format == DataFormat.WKW


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_create_default_mag(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag_view = layer.add_mag("1")

    assert layer.data_format == data_format
    assert mag_view.info.chunk_size == Vec3Int.full(32)
    if data_format == DataFormat.WKW:
        assert mag_view.info.chunks_per_shard == Vec3Int.full(32)
    else:
        assert mag_view.info.chunks_per_shard == Vec3Int.full(1)
    assert mag_view.info.num_channels == 1
    assert mag_view.info.compression_mode == False


def test_create_dataset_with_explicit_header_fields() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)

    ds = Dataset(ds_path, scale=(1, 1, 1))
    ds.add_layer("color", COLOR_CATEGORY, dtype_per_layer="uint48", num_channels=3)

    ds.get_layer("color").add_mag("1", chunk_size=64, chunks_per_shard=64)
    ds.get_layer("color").add_mag("2-2-1")

    assert (ds_path / "color" / "1" / "header.wkw").exists()
    assert (ds_path / "color" / "2-2-1" / "header.wkw").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assert ds.get_layer("color").dtype_per_channel == np.dtype("uint16")
    assert ds.get_layer("color")._properties.element_class == "uint48"
    assert ds.get_layer("color").get_mag(1).info.chunk_size == Vec3Int.full(64)
    assert ds.get_layer("color").get_mag(1).info.chunks_per_shard == Vec3Int.full(64)
    assert ds.get_layer("color").get_mag(1)._properties.cube_length == 64 * 64
    assert ds.get_layer("color").get_mag("2-2-1").info.chunk_size == Vec3Int.full(
        32
    )  # defaults are used
    assert ds.get_layer("color").get_mag("2-2-1").info.chunks_per_shard == Vec3Int.full(
        32
    )  # defaults are used
    assert ds.get_layer("color").get_mag("2-2-1")._properties.cube_length == 32 * 32

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_open_dataset(data_format: DataFormat, output_path: Path) -> None:
    new_dataset_path = copy_simple_dataset(data_format, output_path)
    ds = Dataset.open(new_dataset_path)

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1
    assert ds.get_layer("color").data_format == data_format


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_modify_existing_dataset(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds1 = Dataset(ds_path, scale=(1, 1, 1))
    ds1.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="float",
        num_channels=1,
        data_format=data_format,
    )

    ds2 = Dataset.open(ds_path)

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        "uint8",
        largest_segment_id=100000,
        data_format=data_format,
    ).add_mag("1")

    assert (ds_path / "segmentation" / "1").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_read(data_format: DataFormat, output_path: Path) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)

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
def test_view_write(data_format: DataFormat, output_path: Path) -> None:
    ds_path = copy_simple_dataset(data_format, output_path)
    wk_view = (
        Dataset.open(ds_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    assert wk_view.info.data_format == data_format

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

    wk_view.write(write_data)

    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert np.array_equal(data, write_data)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_view_write_out_of_bounds(data_format: DataFormat, output_path: Path) -> None:
    ds_path = copy_simple_dataset(
        data_format, output_path, "view_dataset_out_of_bounds"
    )

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
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert mag_view.info.data_format == data_format

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_mag_view_write_out_of_bounds_mag2(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "dataset_out_of_bounds")

    ds = Dataset.open(ds_path)
    color_layer = ds.get_layer("color")
    mag_view = color_layer.get_or_add_mag("2-2-1")

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(24, 24, 24)
    mag_view.write(
        np.zeros((3, 50, 1, 48), dtype=np.uint8), absolute_offset=(20, 20, 10)
    )  # this is bigger than the bounding_box
    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)
    assert color_layer.bounding_box.size == Vec3Int(120, 24, 58)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_views_are_equal(data_format: DataFormat, output_path: Path) -> None:
    path_a = prepare_dataset_path(data_format, output_path / "a")
    path_b = prepare_dataset_path(data_format, output_path / "b")
    mag_a = (
        Dataset(path_a, scale=(1, 1, 1))
        .get_or_add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .get_or_add_mag("1")
    )
    mag_b = (
        Dataset(path_b, scale=(1, 1, 1))
        .get_or_add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .get_or_add_mag("1")
    )

    data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)

    mag_a.write(data)
    mag_b.write(data)
    assert mag_a.content_is_equal(mag_b)

    mag_b.write(data + 10)
    assert not mag_a.content_is_equal(mag_b)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_update_new_bounding_box_offset(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = color_layer.add_mag("1")

    assert color_layer.bounding_box.topleft == Vec3Int(0, 0, 0)

    np.random.seed(1234)
    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(
        write_data, absolute_offset=(10, 10, 10)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(10, 10, 10)
    assert color_layer.bounding_box.size == Vec3Int(10, 10, 10)

    mag.write(
        write_data, absolute_offset=(5, 5, 20)
    )  # the write method of MagDataset does always use the relative offset to (0, 0, 0)
    assert color_layer.bounding_box.topleft == Vec3Int(5, 5, 10)
    assert color_layer.bounding_box.size == Vec3Int(15, 15, 20)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_write_multi_channel_uint8(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=data_format
    ).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    mag.write(data)

    assert np.array_equal(data, mag.read())

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_wkw_write_multi_channel_uint16(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "multichannel")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=3,
        dtype_per_layer="uint48",
        data_format=data_format,
    ).add_mag("1")

    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read()

    assert np.array_equal(data, written_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_empty_read(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer("color", category=COLOR_CATEGORY, data_format=data_format)
        .add_mag("1")
    )
    with pytest.raises(AssertionError):
        # size
        mag.read(absolute_offset=(0, 0, 0), size=(0, 0, 0))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_padded_data(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "empty")
    mag = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer(
            "color", category=COLOR_CATEGORY, num_channels=3, data_format=data_format
        )
        .add_mag("1")
    )
    # there is no data yet, however, this should not fail but pad the data with zeros
    data = mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))

    assert data.shape == (3, 10, 10, 10)
    assert np.array_equal(data, np.zeros((3, 10, 10, 10)))


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_num_channel_mismatch_assertion(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", category=COLOR_CATEGORY, num_channels=1, data_format=data_format
    ).add_mag(
        "1"
    )  # num_channel=1 is also the default

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)  # 3 channels

    with pytest.raises(AssertionError):
        mag.write(write_data)  # there is a mismatch between the number of channels

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))

    assert "color" not in ds.layers.keys()

    # layer did not exist before
    layer = ds.get_or_add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_layer="uint8",
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
        dtype_per_layer="uint8",
        num_channels=1,
        data_format=data_format,
    )
    assert "color" in ds.layers.keys()
    assert layer.name == "color"
    assert layer.data_format == data_format

    with pytest.raises(AssertionError):
        # The layer "color" did exist before but with another 'dtype_per_layer' (this would work the same for 'category' and 'num_channels')
        ds.get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_layer="uint16",
            num_channels=1,
            data_format=data_format,
        )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_layer_idempotence(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    ds.get_or_add_layer(
        "color2", category="color", dtype_per_channel=np.uint8, data_format=data_format
    ).get_or_add_mag("1")
    ds.get_or_add_layer(
        "color2", category="color", dtype_per_channel=np.uint8, data_format=data_format
    ).get_or_add_mag("1")

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_get_or_add_mag(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)

    layer = Dataset(ds_path, scale=(1, 1, 1)).add_layer(
        "color", category=COLOR_CATEGORY, data_format=data_format
    )

    assert Mag(1) not in layer.mags.keys()

    chunk_size, chunks_per_shard = default_chunk_config(data_format, 32)

    # The mag did not exist before
    mag = layer.get_or_add_mag(
        "1",
        chunk_size=chunk_size,
        chunks_per_shard=chunks_per_shard,
        compress=False,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    # The mag did exist before
    layer.get_or_add_mag(
        "1",
        chunk_size=chunk_size,
        chunks_per_shard=chunks_per_shard,
        compress=False,
    )
    assert Mag(1) in layer.mags.keys()
    assert mag.name == "1"
    assert mag.info.data_format == data_format

    with pytest.raises(AssertionError):
        # The mag "1" did exist before but with another 'chunk_size' (this would work the same for 'chunks_per_shard' and 'compress')
        layer.get_or_add_mag(
            "1",
            chunk_size=Vec3Int.full(64),
            chunks_per_shard=chunks_per_shard,
            compress=False,
        )

    assure_exported_properties(layer.dataset)


def test_open_dataset_without_num_channels_in_properties() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "old_wkw")
    copytree(TESTDATA_DIR / "old_wkw_dataset", ds_path)

    with open(
        ds_path / "datasource-properties.json",
        encoding="utf-8",
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("num_channels") is None

    ds = Dataset.open(ds_path)
    assert ds.get_layer("color").num_channels == 1
    ds._export_as_json()

    with open(
        ds_path / "datasource-properties.json",
        encoding="utf-8",
    ) as datasource_properties:
        data = json.load(datasource_properties)
        assert data["dataLayers"][0].get("numChannels") == 1

    assure_exported_properties(ds)


def test_largest_segment_id_requirement() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(10, 10, 10))

    with pytest.raises(AssertionError):
        ds.add_layer("segmentation", SEGMENTATION_CATEGORY)

    largest_segment_id = 10
    ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        largest_segment_id=largest_segment_id,
    ).add_mag(Mag(1))

    ds = Dataset.open(ds_path)
    assert (
        cast(SegmentationLayer, ds.get_layer("segmentation")).largest_segment_id
        == largest_segment_id
    )

    assure_exported_properties(ds)


def test_properties_with_segmentation() -> None:
    ds_path = prepare_dataset_path(
        DataFormat.WKW, TESTOUTPUT_DIR, "complex_property_ds"
    )
    copytree(TESTDATA_DIR / "complex_property_ds", ds_path)

    with open(ds_path / "datasource-properties.json", "r", encoding="utf-8") as f:
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

    with open(ds_path / "datasource-properties.json", "w", encoding="utf-8") as f:
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
            ds_path / "datasource-properties.json", "r", encoding="utf-8"
        ) as output_properties:
            output_data = json.load(output_properties)
            for layer in output_data["dataLayers"]:
                # remove the num_channels because they are not part of the original json
                if "numChannels" in layer:
                    del layer["numChannels"]

            assert input_data == output_data


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wk(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(2, 2, 1))
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)

    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(
        "1",
        chunks_per_shard=chunks_per_shard,
        chunk_size=chunk_size,
    )

    original_data = (np.random.rand(50, 100, 150) * 205).astype(np.uint8)
    mag.write(absolute_offset=(70, 80, 90), data=original_data)

    # Test with executor
    with get_executor_for_args(None) as executor:
        mag.for_each_chunk(
            chunk_job,
            chunk_size=(64, 64, 64),
            executor=executor,
        )
    assert np.array_equal(original_data + 50, mag.get_view().read()[0])

    # Reset the data
    mag.write(absolute_offset=(70, 80, 90), data=original_data)

    # Test without executor
    mag.for_each_chunk(
        chunk_job,
        chunk_size=(64, 64, 64),
    )
    assert np.array_equal(original_data + 50, mag.get_view().read()[0])

    assure_exported_properties(ds)


# Don't test zarr for performance reasons (lack of sharding)
def test_chunking_wkw_advanced() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "chunking_advanced")
    ds = Dataset(ds_path, scale=(1, 1, 2))

    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
    ).add_mag(
        "1",
        chunk_size=8,
        chunks_per_shard=8,
    )
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view(absolute_offset=(10, 10, 10), size=(150, 150, 54))
    for_each_chunking_advanced(ds, view)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_chunking_wkw_wrong_chunk_size(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "chunking_with_wrong_chunk_size"
    )
    ds = Dataset(ds_path, scale=(1, 1, 2))
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag(
        "1",
        chunk_size=chunk_size,
        chunks_per_shard=chunks_per_shard,
    )
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    view = mag.get_view()

    for_each_chunking_with_wrong_chunk_size(view)

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
    ds1 = Dataset(ds_path, scale=(1, 1, 1), exist_ok=False)
    assert "color" not in ds1.layers.keys()
    ds1.add_layer("color", COLOR_CATEGORY)
    assert "color" in ds1.layers.keys()

    # dataset already exists
    ds2 = Dataset(ds_path, scale=(1, 1, 1), exist_ok=True)
    assert "color" in ds2.layers.keys()

    ds2 = Dataset(ds_path, scale=(1, 1, 1), name="wkw_dataset_exist_ok", exist_ok=True)
    assert "color" in ds2.layers.keys()

    with pytest.raises(AssertionError):
        # dataset already exists, but with a different scale
        Dataset(ds_path, scale=(2, 2, 2), exist_ok=True)

    with pytest.raises(AssertionError):
        # dataset already exists, but with a different name
        Dataset(ds_path, scale=(1, 1, 1), name="some different name", exist_ok=True)

    assure_exported_properties(ds1)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_changing_layer_bounding_box(
    data_format: DataFormat, output_path: Path
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
    assert np.array_equal(original_data[:, :12, :12, :10], less_data)

    layer.bounding_box = layer.bounding_box.with_size(
        [36, 48, 60]
    )  # increase the bounding box

    bbox_size = ds.get_layer("color").bounding_box.size
    assert tuple(bbox_size) == (36, 48, 60)
    more_data = mag.read(absolute_offset=(0, 0, 0), size=bbox_size)
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
    assert np.array_equal(
        original_data,
        mag.read(absolute_offset=(0, 0, 0), size=mag.bounding_box.bottomright),
    )

    assert np.array_equal(
        original_data[:, 10:, 10:, :],
        mag.read(absolute_offset=(10, 10, 0), size=(14, 14, 24)),
    )

    # resetting the offset to (0, 0, 0)
    # Note that the size did not change. Therefore, the new bottom right is now at (14, 14, 24)
    layer.bounding_box = BoundingBox((0, 0, 0), new_bbox_size)
    new_data = mag.read()
    assert new_data.shape == (3, 14, 14, 24)
    assert np.array_equal(original_data[:, :14, :14, :], new_data)

    assure_exported_properties(ds)


def test_get_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "get_view")
    ds = Dataset(ds_path, scale=(1, 1, 1))
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
    mag.write(write_data, absolute_offset=(10, 20, 30))

    with pytest.raises(AssertionError):
        # The offset and size default to (0, 0, 0).
        # Sizes that contain "0" are not allowed
        mag.get_view(absolute_offset=(0, 0, 0), size=(10, 10, 0))

    assert mag.bounding_box.bottomright == Vec3Int(110, 220, 330)

    # Therefore, creating a view with a size of (16, 16, 16) is now allowed
    wk_view = mag.get_view(relative_offset=(0, 0, 0), size=(16, 16, 16))
    assert wk_view.bounding_box == BoundingBox((10, 20, 30), (16, 16, 16))

    with pytest.raises(AssertionError):
        # Creating this view does not work because the offset (0, 0, 0) would be outside
        # of the bounding box from the properties.json.
        mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0))

    # But setting "read_only=True" still works
    mag.get_view(size=(26, 36, 46), absolute_offset=(0, 0, 0), read_only=True)

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
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "invalid_dtype")
    ds = Dataset(ds_path, scale=(1, 1, 1))
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
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "valid_dtype")
    ds = Dataset(ds_path, scale=(1, 1, 1))
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
        ds_path / "datasource-properties.json",
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
        ds_path
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_multi_channel(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = (np.random.rand(3, 100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, num_channels=3, data_format=data_format)
        .add_mag(
            "1", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard, compress=True
        )
    )
    mag_view.write(write_data1)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        write_data2 = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
        # Writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
        # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
        compressed_mag.get_view(relative_offset=(50, 60, 70), size=(50, 60, 70)).write(
            relative_offset=(10, 20, 30), data=write_data2
        )

    assert np.array_equal(
        write_data2,
        compressed_mag.read(relative_offset=(60, 80, 100), size=(10, 10, 10)),
    )  # the new data was written
    assert np.array_equal(
        write_data1[:, :60, :80, :100],
        compressed_mag.read(relative_offset=(0, 0, 0), size=(60, 80, 100)),
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data_single_channel(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 120, 140) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard, compress=True
        )
    )
    mag_view.write(write_data1)

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        write_data2 = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
        # Writing unaligned data to a compressed dataset works because the data gets padded, but it prints a warning
        # Writing compressed data directly to "compressed_mag" also works, but using a View here covers an additional edge case
        compressed_mag.get_view(absolute_offset=(50, 60, 70), size=(50, 60, 70)).write(
            relative_offset=(10, 20, 30), data=write_data2
        )

    assert np.array_equal(
        write_data2,
        compressed_mag.read(absolute_offset=(60, 80, 100), size=(10, 10, 10))[0],
    )  # the new data was written
    assert np.array_equal(
        write_data1[:60, :80, :100],
        compressed_mag.read(absolute_offset=(0, 0, 0), size=(60, 80, 100))[0],
    )  # the old data is still there


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_compressed_data(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    mag_view = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "2", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard, compress=True
        )
    )
    mag_view.write((np.random.rand(120, 140, 160) * 255).astype(np.uint8))

    # open compressed dataset
    compressed_mag = Dataset.open(ds_path).get_layer("color").get_mag("2")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.
        compressed_mag.write(
            absolute_offset=(10, 20, 30),
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_writing_subset_of_chunked_compressed_data(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "compressed_data")
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)

    # create uncompressed dataset
    write_data1 = (np.random.rand(100, 200, 300) * 255).astype(np.uint8)
    mag_view = (
        Dataset(ds_path, scale=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY, data_format=data_format)
        .add_mag(
            "1", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard, compress=True
        )
    )
    mag_view.write(write_data1)

    # open compressed dataset
    compressed_view = (
        Dataset.open(ds_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(100, 200, 300))
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="webknossos"
        )  # This line is not necessary. It simply keeps the output of the tests clean.

        # Easy case:
        # The aligned data (offset=(0,0,0), size=(64, 64, 64)) IS fully within the bounding box of the view
        write_data2 = (np.random.rand(50, 40, 30) * 255).astype(np.uint8)
        compressed_view.write(absolute_offset=(10, 20, 30), data=write_data2)

        # Advanced case:
        # The aligned data (offset=(0,0,0), size=(128, 128, 128)) is NOT fully within the bounding box of the view
        compressed_view.write(
            absolute_offset=(10, 20, 30),
            data=(np.random.rand(90, 80, 70) * 255).astype(np.uint8),
        )

    np.array_equal(
        write_data2,
        compressed_view.read(absolute_offset=(10, 20, 30), size=(50, 40, 30)),
    )  # the new data was written
    np.array_equal(
        write_data1[:10, :20, :30],
        compressed_view.read(absolute_offset=(0, 0, 0), size=(10, 20, 30)),
    )  # the old data is still there


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_add_symlink_layer(data_format: DataFormat) -> None:
    ds_path = copy_simple_dataset(data_format, TESTOUTPUT_DIR, "original")
    symlink_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "with_symlink")

    # Add an additional segmentation layer to the original dataset
    Dataset.open(ds_path).add_layer(
        "segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999
    )

    original_mag = Dataset.open(ds_path).get_layer("color").get_mag("1")

    ds = Dataset(symlink_path, scale=(1, 1, 1))
    # symlink color layer
    symlink_layer = ds.add_symlink_layer(ds_path / "color")
    # symlink segmentation layer
    symlink_segmentation_layer = ds.add_symlink_layer(ds_path / "segmentation")
    mag = symlink_layer.get_mag("1")

    assert (symlink_path / "color" / "1").exists()
    assert (symlink_path / "segmentation").exists()

    assert len(ds.layers) == 2
    assert len(ds.get_layer("color").mags) == 1

    assert cast(SegmentationLayer, symlink_segmentation_layer).largest_segment_id == 999

    # write data in symlink layer
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data)

    assert np.array_equal(
        mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)), write_data
    )
    assert np.array_equal(
        original_mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10)), write_data
    )

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format", DATA_FORMATS)
def test_add_symlink_mag(data_format: DataFormat) -> None:
    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "original")
    symlink_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "with_symlink")

    original_ds = Dataset(ds_path, scale=(1, 1, 1))
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

    ds = Dataset(symlink_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, dtype_per_channel="uint8")
    layer.add_mag(1).write(
        absolute_offset=(6, 6, 6),
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8),
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    symlink_mag_2 = layer.add_symlink_mag(original_mag_2)
    layer.add_symlink_mag(original_mag_4.path)

    assert (symlink_path / "color" / "1").exists()
    assert len(layer._properties.mags) == 3

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    # Write data in symlink layer
    # Note: The written data is fully inside the bounding box of the original data.
    # This is important because the bounding box of the foreign layer would not be updated if we use the linked dataset to write outside of its original bounds.
    write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    symlink_mag_2.write(absolute_offset=(0, 0, 0), data=write_data)

    assert np.array_equal(
        symlink_mag_2.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0], write_data
    )
    assert np.array_equal(
        original_layer.get_mag(2).read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0],
        write_data,
    )

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


def test_remote_add_symlink_layer() -> None:
    src_dataset_path = copy_simple_dataset(DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR)
    dst_dataset_path = prepare_dataset_path(
        DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR, "with_symlink"
    )

    src_ds = Dataset.open(src_dataset_path)
    dst_ds = Dataset(dst_dataset_path, scale=(1, 1, 1))

    with pytest.raises(AssertionError):
        dst_ds.add_symlink_layer(src_ds.get_layer("color"))


def test_remote_add_symlink_mag() -> None:
    src_dataset_path = copy_simple_dataset(DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR)
    dst_dataset_path = prepare_dataset_path(
        DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR, "with_symlink"
    )

    src_ds = Dataset.open(src_dataset_path)
    src_layer = src_ds.get_layer("color")
    src_mag1 = src_layer.get_mag("1")

    dst_ds = Dataset(dst_dataset_path, scale=(1, 1, 1))
    dst_layer = dst_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=DataFormat.Zarr
    )

    with pytest.raises(AssertionError):
        dst_layer.add_symlink_mag(src_mag1)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_copy_mag(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    symlink_path = prepare_dataset_path(data_format, output_path, "with_symlink")

    original_ds = Dataset(ds_path, scale=(1, 1, 1))
    original_layer = original_ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=data_format
    )
    original_layer.add_mag(1).write(
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    )
    original_data = (np.random.rand(5, 10, 15) * 255).astype(np.uint8)
    original_mag_2 = original_layer.add_mag(2)
    original_mag_2.write(data=original_data)

    ds = Dataset(symlink_path, scale=(1, 1, 1))
    layer = ds.add_layer(
        "color", COLOR_CATEGORY, dtype_per_channel="uint8", data_format=data_format
    )
    layer.add_mag(1).write(
        absolute_offset=(6, 6, 6),
        data=(np.random.rand(10, 20, 30) * 255).astype(np.uint8),
    )

    assert tuple(layer.bounding_box.topleft) == (6, 6, 6)
    assert tuple(layer.bounding_box.size) == (10, 20, 30)

    copy_mag = layer.add_copy_mag(original_mag_2)

    assert (symlink_path / "color" / "1").exists()
    assert len(layer._properties.mags) == 2

    assert tuple(layer.bounding_box.topleft) == (0, 0, 0)
    assert tuple(layer.bounding_box.size) == (16, 26, 36)

    # Write data in copied layer
    write_data = (np.random.rand(5, 5, 5) * 255).astype(np.uint8)
    copy_mag.write(absolute_offset=(0, 0, 0), data=write_data)

    assert np.array_equal(
        copy_mag.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))[0], write_data
    )
    assert np.array_equal(original_layer.get_mag(2).read()[0], original_data)

    assure_exported_properties(ds)
    assure_exported_properties(original_ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_search_dataset_also_for_long_layer_name(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "long_layer_name")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format).add_mag("2")

    assert mag.name == "2"
    short_mag_file_path = ds.path / "color" / Mag(mag.name).to_layer_name()
    long_mag_file_path = ds.path / "color" / Mag(mag.name).to_long_layer_name()

    assert short_mag_file_path.exists()
    assert not long_mag_file_path.exists()

    write_data = (np.random.rand(10, 10, 10) * 255).astype(np.uint8)
    mag.write(write_data, absolute_offset=(20, 20, 20))

    assert np.array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )

    # rename the path from "long_layer_name/color/2" to "long_layer_name/color/2-2-2"
    copytree(short_mag_file_path, long_mag_file_path)
    rmtree(short_mag_file_path)

    # make sure that reading data still works
    mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20))

    # when opening the dataset, it searches both for the long and the short path
    layer = Dataset.open(ds_path).get_layer("color")
    mag = layer.get_mag("2")
    assert np.array_equal(
        mag.read(absolute_offset=(20, 20, 20), size=(20, 20, 20)),
        np.expand_dims(write_data, 0),
    )
    layer.delete_mag("2")

    # Note: 'ds' is outdated (it still contains Mag(2)) because it was opened again and changed.
    assure_exported_properties(layer.dataset)


def test_outdated_dtype_parameter() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "outdated_dtype")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    with pytest.raises(ValueError):
        ds.get_or_add_layer("color", COLOR_CATEGORY, dtype=np.uint8, num_channels=1)

    with pytest.raises(ValueError):
        ds.add_layer("color", COLOR_CATEGORY, dtype=np.uint8, num_channels=1)


@pytest.mark.parametrize("make_relative", [True, False])
@pytest.mark.parametrize(
    "data_format", DATA_FORMATS
)  # Cannot test symlinks on remote storage
def test_dataset_shallow_copy(make_relative: bool, data_format: DataFormat) -> None:
    print(make_relative, data_format, TESTOUTPUT_DIR)

    ds_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "original")
    copy_path = prepare_dataset_path(data_format, TESTOUTPUT_DIR, "copy")

    ds = Dataset(ds_path, (1, 1, 1))
    original_layer_1 = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer=np.uint8,
        num_channels=1,
        data_format=data_format,
    )
    original_layer_1.add_mag(1)
    original_layer_1.add_mag("2-2-1")
    original_layer_2 = ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        dtype_per_layer=np.uint32,
        largest_segment_id=0,
        data_format=data_format,
    )
    original_layer_2.add_mag(4)
    mappings_path = original_layer_2.path / "mappings"
    mappings_path.mkdir(parents=True)
    open(mappings_path / "agglomerate_view.hdf5", "w", encoding="utf-8").close()

    shallow_copy_of_ds = ds.shallow_copy_dataset(copy_path, make_relative=make_relative)
    shallow_copy_of_ds.get_layer("color").add_mag(Mag("4-4-1"))
    assert (
        len(Dataset.open(ds_path).get_layer("color").mags) == 2
    ), "Adding a new mag should not affect the original dataset"
    assert (
        len(Dataset.open(copy_path).get_layer("color").mags) == 3
    ), "Expecting all mags from original dataset and new downsampled mag"
    assert (
        copy_path / "segmentation" / "mappings" / "agglomerate_view.hdf5"
    ).exists(), "Expecting mappings to exist in shallow copy"


def test_remote_wkw_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    with pytest.raises(AssertionError):
        ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.WKW)


def test_dataset_conversion_wkw_only() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "original")
    converted_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "converted")

    # create example dataset
    origin_ds = Dataset(ds_path, scale=(1, 1, 1))
    seg_layer = origin_ds.add_layer(
        "layer1",
        SEGMENTATION_CATEGORY,
        num_channels=1,
        largest_segment_id=1000000000,
    )
    seg_layer.add_mag(
        "1", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(16)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(128, 128, 256) * 255).astype(np.uint8),
    )
    seg_layer.add_mag(
        "2", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(16)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(64, 64, 128) * 255).astype(np.uint8),
    )
    wk_color_layer = origin_ds.add_layer("layer2", COLOR_CATEGORY, num_channels=3)
    wk_color_layer.add_mag(
        "1", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(16)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 128, 128, 256) * 255).astype(np.uint8),
    )
    wk_color_layer.add_mag(
        "2", chunk_size=Vec3Int.full(8), chunks_per_shard=Vec3Int.full(16)
    ).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 64, 64, 128) * 255).astype(np.uint8),
    )
    converted_ds = origin_ds.copy_dataset(converted_path)

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
            assert origin_info.data_format == converted_info.data_format
            assert np.array_equal(
                origin_ds.layers[layer_name].mags[mag].read(),
                converted_ds.layers[layer_name].mags[mag].read(),
            )

    assure_exported_properties(origin_ds)
    assure_exported_properties(converted_ds)


@pytest.mark.parametrize("output_path", [TESTOUTPUT_DIR, REMOTE_TESTOUTPUT_DIR])
def test_dataset_conversion_from_wkw_to_zarr(output_path: Path) -> None:
    converted_path = prepare_dataset_path(DataFormat.Zarr, output_path, "converted")

    input_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")
    converted_ds = input_ds.copy_dataset(
        converted_path, data_format=DataFormat.Zarr, chunks_per_shard=1
    )

    assert (converted_path / "color" / "1" / ".zarray").exists()
    assert np.all(
        input_ds.get_layer("color").get_mag("1").read()
        == converted_ds.get_layer("color").get_mag("1").read()
    )
    assert converted_ds.get_layer("color").data_format == DataFormat.Zarr
    assert (
        converted_ds.get_layer("color").get_mag("1").info.data_format == DataFormat.Zarr
    )

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

    ds = Dataset(src_dataset_path, scale=(1, 1, 2))
    mag = ds.add_layer(
        "color",
        category=COLOR_CATEGORY,
        dtype_per_channel="uint8",
        num_channels=3,
        data_format=data_format,
    ).add_mag("1")
    mag.write(data=(np.random.rand(3, 256, 256, 256) * 255).astype(np.uint8))
    source_view = mag.get_view(absolute_offset=(0, 0, 0), size=(256, 256, 256))

    target_mag = (
        Dataset(dst_dataset_path, scale=(1, 1, 2))
        .get_or_add_layer(
            "color",
            COLOR_CATEGORY,
            dtype_per_channel="uint8",
            num_channels=3,
            data_format=data_format,
        )
        .get_or_add_mag(
            "1",
            chunk_size=Vec3Int.full(8),
            chunks_per_shard=(4 if data_format == DataFormat.WKW else 1),
        )
    )

    target_mag.layer.bounding_box = BoundingBox((0, 0, 0), (256, 256, 256))
    target_view = target_mag.get_view(absolute_offset=(0, 0, 0), size=(256, 256, 256))

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
        source_view.read() + 50,
        target_view.read(),
    )

    assure_exported_properties(ds)


def _func_invalid_target_chunk_size_wk(args: Tuple[View, View, int]) -> None:
    (_s, _t, _i) = args


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_for_zipped_chunks_invalid_target_chunk_size_wk(
    data_format: DataFormat, output_path: Path
) -> None:
    ds_path = prepare_dataset_path(
        data_format, output_path, "zipped_chunking_source_invalid"
    )
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)
    test_cases_wk = [
        (10, 20, 30),
        (64, 64, 100),
        (64, 50, 64),
        (200, 128, 128),
    ]

    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer1 = ds.get_or_add_layer("color1", COLOR_CATEGORY, data_format=data_format)
    source_mag_view = layer1.get_or_add_mag(
        1, chunk_size=chunk_size, chunks_per_shard=chunks_per_shard
    )

    layer2 = ds.get_or_add_layer("color2", COLOR_CATEGORY, data_format=data_format)
    target_mag_view = layer2.get_or_add_mag(
        1, chunk_size=chunk_size, chunks_per_shard=chunks_per_shard
    )

    source_view = source_mag_view.get_view(
        absolute_offset=(0, 0, 0), size=(300, 300, 300), read_only=True
    )
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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_read_only_view(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "read_only_view")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag = ds.get_or_add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    ).get_or_add_mag("1")
    mag.write(
        data=(np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8),
        absolute_offset=(10, 20, 30),
    )
    v_write = mag.get_view()
    v_read = mag.get_view(read_only=True)

    new_data = (np.random.rand(1, 5, 6, 7) * 255).astype(np.uint8)
    with pytest.raises(AssertionError):
        v_read.write(data=new_data)

    v_write.write(data=new_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_bounding_box_on_disk(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(2, 2, 1))
    chunk_size, chunks_per_shard = default_chunk_config(data_format, 8)
    mag = ds.add_layer("color", category="color", data_format=data_format).add_mag(
        "2-2-1", chunk_size=chunk_size, chunks_per_shard=chunks_per_shard
    )  # cube_size = 8*8 = 64

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


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_compression(data_format: DataFormat, output_path: Path) -> None:
    new_dataset_path = copy_simple_dataset(data_format, output_path)
    mag1 = Dataset.open(new_dataset_path).get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100))

    assert not mag1._is_compressed()

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Remote datasets require a `target_path` for compression
        with pytest.raises(AssertionError):
            mag1.compress()

        compressed_dataset_path = (
            REMOTE_TESTOUTPUT_DIR / "simple_zarr_dataset_compressed"
        )
        mag1.compress(
            target_path=compressed_dataset_path,
        )
        mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    else:
        mag1.compress()

    assert mag1._is_compressed()
    assert mag1.info.data_format == data_format

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


def test_dataset_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, scale=(2, 2, 1))
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
    with open(ds1.path / PROPERTIES_FILE_NAME, encoding="utf-8") as f:
        properties = json.load(f)
        view_configuration_dict = properties["defaultViewConfiguration"]
        for k in view_configuration_dict.keys():
            assert snake_to_camel_case(k) == k

    assure_exported_properties(ds1)


def test_layer_view_configuration() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds1 = Dataset(ds_path, scale=(2, 2, 1))
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
    ds2 = Dataset.open(ds_path)
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


def test_get_largest_segment_id() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))

    segmentation_layer = cast(
        SegmentationLayer,
        ds.add_layer("segmentation", SEGMENTATION_CATEGORY, largest_segment_id=999),
    )
    assert segmentation_layer.largest_segment_id == 999
    segmentation_layer.largest_segment_id = 123
    assert segmentation_layer.largest_segment_id == 123

    assure_exported_properties(ds)


def test_get_or_add_layer_by_type() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))
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
    ds = Dataset(ds_path / "some_name", scale=(1, 1, 1))
    assert ds.name == "some_name"
    ds.name = "other_name"
    assert ds.name == "other_name"

    ds2 = Dataset(
        ds_path / "some_new_name", scale=(1, 1, 1), name="very important dataset"
    )
    assert ds2.name == "very important dataset"

    assure_exported_properties(ds)


def test_read_bbox() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(2, 2, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY)
    mag = layer.add_mag(1)
    mag.write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(50, 60, 70) * 255).astype(np.uint8),
    )

    assert np.array_equal(
        mag.read(absolute_offset=(20, 30, 40), size=(40, 50, 60)),
        mag.read(
            absolute_bounding_box=BoundingBox(topleft=(20, 30, 40), size=(40, 50, 60))
        ),
    )


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_copy_layer(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path, "original")
    copy_path = prepare_dataset_path(data_format, output_path, "copy")

    ds = Dataset(ds_path, scale=(2, 2, 1))

    # Create dataset to copy data from
    other_ds = Dataset(copy_path, scale=(2, 2, 1))
    original_color_layer = other_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=data_format
    )
    original_color_layer.add_mag(1).write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(32, 64, 128) * 255).astype(np.uint8),
    )
    other_ds.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        data_format=data_format,
        largest_segment_id=999,
    ).add_mag("1")

    # Copies the "color" layer from a different dataset
    ds.add_copy_layer(copy_path / "color")
    ds.add_copy_layer(copy_path / "segmentation")
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
        ds.add_copy_layer(copy_path / "color")

    # Test if the changes of the properties are persisted on disk by opening it again
    assert "color" in Dataset.open(ds_path).layers.keys()

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_rename_layer(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=data_format)
    mag = layer.add_mag(1)
    write_data = (np.random.rand(10, 20, 30) * 255).astype(np.uint8)
    mag.write(data=write_data)

    if output_path == REMOTE_TESTOUTPUT_DIR:
        # Cannot rename layers on remote storage
        with pytest.raises(AssertionError):
            layer.name = "color2"
        return
    else:
        layer.name = "color2"

    assert not (ds_path / "color").exists()
    assert (ds_path / "color2").exists()
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 0
    assert len([l for l in ds._properties.data_layers if l.name == "color2"]) == 1
    assert "color2" in ds.layers.keys()
    assert "color" not in ds.layers.keys()
    assert ds.get_layer("color2").data_format == data_format

    # The "mag" object which was created before renaming the layer is still valid
    assert np.array_equal(mag.read()[0], write_data)

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_delete_layer_and_mag(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
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
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 1
    assert len([l for l in ds._properties.data_layers if l.name == "segmentation"]) == 1
    assert len(color_layer._properties.mags) == 2

    color_layer.delete_mag(1)
    assert len(color_layer._properties.mags) == 1
    assert len([m for m in color_layer._properties.mags if Mag(m.mag) == Mag(2)]) == 1

    ds.delete_layer("color")
    assert "color" not in ds.layers
    assert "segmentation" in ds.layers
    assert len([l for l in ds._properties.data_layers if l.name == "color"]) == 0
    assert len([l for l in ds._properties.data_layers if l.name == "segmentation"]) == 1

    assure_exported_properties(ds)


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_add_layer_like(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds = Dataset(ds_path, scale=(1, 1, 1))
    color_layer1 = ds.add_layer(
        "color1",
        COLOR_CATEGORY,
        dtype_per_layer="uint24",
        num_channels=3,
        data_format=data_format,
    )
    color_layer1.add_mag(1)
    segmentation_layer1 = cast(
        SegmentationLayer,
        ds.add_layer(
            "segmentation1",
            SEGMENTATION_CATEGORY,
            dtype_per_channel="uint8",
            largest_segment_id=999,
            data_format=data_format,
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


def test_pickle_view() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR, "pickle")
    ds = Dataset(ds_path, scale=(1, 1, 1))
    mag1 = ds.add_layer("color", COLOR_CATEGORY).add_mag(1)

    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag1.write(data_to_write)
    assert mag1._cached_array is not None

    with (ds_path / "save.p").open("wb") as f_write:
        pickle.dump(mag1, f_write)
    with (ds_path / "save.p").open("rb") as f_read:
        pickled_mag1 = pickle.load(f_read)

    # Make sure that the pickled mag can still read data
    assert pickled_mag1._cached_array is None
    assert np.array_equal(
        data_to_write,
        pickled_mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )
    assert pickled_mag1._cached_array is not None

    # Make sure that the attributes of the MagView (not View) still exist
    assert pickled_mag1.layer is not None


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_warn_outdated_properties(data_format: DataFormat, output_path: Path) -> None:
    ds_path = prepare_dataset_path(data_format, output_path)
    ds1 = Dataset(ds_path, scale=(1, 1, 1))
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


def test_can_compress_mag8() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, TESTOUTPUT_DIR)
    ds = Dataset(ds_path, scale=(1, 1, 1))

    layer = ds.add_layer("color", COLOR_CATEGORY)
    layer.bounding_box = BoundingBox((0, 0, 0), (12240, 12240, 685))
    for mag in ["1", "2-2-1", "4-4-1", "8-8-2"]:
        layer.add_mag(mag)

    assert layer.bounding_box == BoundingBox((0, 0, 0), (12240, 12240, 685))

    mag_view = layer.get_mag("8-8-2")
    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag_view.write(data_to_write, absolute_offset=(11264, 11264, 0))
    mag_view.compress()


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_downsampling(data_format: DataFormat, output_path: Path) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "downsampling")

    color_layer = Dataset.open(ds_path).get_layer("color")
    color_layer.downsample()

    assert (ds_path / "color" / "2").exists()
    assert (ds_path / "color" / "4").exists()

    if data_format == DataFormat.Zarr:
        assert (ds_path / "color" / "2" / ".zarray").exists()
        assert (ds_path / "color" / "4" / ".zarray").exists()
    else:
        assert (ds_path / "color" / "2" / "header.wkw").exists()
        assert (ds_path / "color" / "4" / "header.wkw").exists()

    assure_exported_properties(color_layer.dataset)


def test_zarr_copy_to_remote_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR, "copied")
    Dataset.open(TESTDATA_DIR / "simple_zarr_dataset").copy_dataset(
        ds_path,
        chunks_per_shard=1,
        data_format=DataFormat.Zarr,
    )
    assert (ds_path / "color" / "1" / ".zarray").exists()


def test_wkw_copy_to_remote_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "copied")
    wkw_ds = Dataset.open(TESTDATA_DIR / "simple_wkw_dataset")

    # Fails with explicit data_format=wkw ...
    with pytest.raises(AssertionError):
        wkw_ds.copy_dataset(ds_path, chunks_per_shard=1, data_format=DataFormat.WKW)

    # ... and with implicit data_format=wkw from the source layers.
    with pytest.raises(AssertionError):
        wkw_ds.copy_dataset(
            ds_path,
            chunks_per_shard=1,
        )

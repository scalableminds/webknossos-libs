import os
import shutil
from typing import Iterator, Tuple

import numpy as np
import pytest
import s3fs
from upath import UPath as Path

from tests.constants import TESTDATA_DIR  # pylint: disable=unused-import
from webknossos.dataset import COLOR_CATEGORY, Dataset
from webknossos.dataset._array import DataFormat
from webknossos.dataset.layer_categories import SEGMENTATION_CATEGORY
from webknossos.geometry import Vec3Int

S3_KEY = "ANTN35UAENTS5UIAEATD"
S3_SECRET = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ENDPOINT = "http://localhost:9000"

BUCKET_PATH = Path(
    "s3://test",
    key=S3_KEY,
    secret=S3_SECRET,
    client_kwargs={"endpoint_url": S3_ENDPOINT},
)


def assure_exported_properties(ds: Dataset) -> None:
    reopened_ds = Dataset.open(ds.path)
    assert (
        ds._properties == reopened_ds._properties
    ), "The properties did not match after reopening the dataset. This might indicate that the properties were not exported after they were changed in memory."


def get_multichanneled_data(dtype: type) -> np.ndarray:
    data = np.zeros((3, 250, 200, 10), dtype=dtype)
    for h in range(10):
        for i in range(250):
            for j in range(200):
                data[0, i, j, h] = i * 256
                data[1, i, j, h] = j * 256
                data[2, i, j, h] = 100 * 256
    return data


def copytree(in_path: Path, out_path: Path) -> None:
    def _walk(path: Path, base_path: Path) -> Iterator[Tuple[Path, Path]]:
        yield (path, path.relative_to(base_path))
        if path.is_dir():
            for p in path.iterdir():
                yield from _walk(p, base_path)
        else:
            yield (path, path.relative_to(base_path))

    for in_sub_path, sub_path in _walk(in_path, in_path):
        if in_sub_path.is_dir():
            (out_path / sub_path).mkdir(parents=True, exist_ok=True)
        else:
            with (in_path / sub_path).open("rb") as in_file, (out_path / sub_path).open(
                "wb"
            ) as out_file:
                shutil.copyfileobj(in_file, out_file)


@pytest.fixture(scope="session", autouse=True)
def create_bucket():
    BUCKET_PATH.fs.mkdirs("test", exist_ok=True)


def delete_dir(relative_path: Path) -> None:
    if relative_path.exists() and relative_path.is_dir():
        relative_path.rmdir(recursive=True)


def test_s3_dataset() -> None:
    delete_dir(BUCKET_PATH / "zarr_dataset")

    ds_path = BUCKET_PATH / "zarr_dataset"
    if ds_path.exists():
        ds_path.rmdir()

    ds = Dataset(ds_path, scale=(1, 1, 1))
    layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr)
    mag1 = layer.add_mag(1)

    data_to_write = (np.random.rand(1, 10, 10, 10) * 255).astype(np.uint8)
    mag1.write(data_to_write)

    assert np.array_equal(
        data_to_write,
        mag1.read(relative_offset=(0, 0, 0), size=data_to_write.shape[-3:]),
    )


def test_create_dataset_with_layer_and_mag() -> None:
    delete_dir(BUCKET_PATH / "zarr_dataset")

    ds = Dataset(BUCKET_PATH / "zarr_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color", data_format=DataFormat.Zarr)

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert (BUCKET_PATH / "zarr_dataset" / "color" / "1").exists()
    assert (BUCKET_PATH / "zarr_dataset" / "color" / "2-2-1").exists()

    assert (BUCKET_PATH / "zarr_dataset" / "color" / "1" / ".zarray").exists()
    assert (BUCKET_PATH / "zarr_dataset" / "color" / "2-2-1" / ".zarray").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assure_exported_properties(ds)


def test_open_dataset() -> None:
    delete_dir(BUCKET_PATH / f"simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / f"simple_zarr_dataset",
        BUCKET_PATH / f"simple_zarr_dataset",
    )
    ds = Dataset.open(BUCKET_PATH / f"simple_zarr_dataset")

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1
    assert ds.get_layer("color").data_format == data_format


def test_modify_existing_dataset() -> None:
    delete_dir(BUCKET_PATH / f"simple_zarr_dataset")
    ds1 = Dataset(BUCKET_PATH / f"simple_zarr_dataset", scale=(1, 1, 1))
    ds1.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="float",
        num_channels=1,
        data_format=DataFormat.Zarr,
    )

    ds2 = Dataset.open(BUCKET_PATH / f"simple_zarr_dataset")

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        "uint8",
        largest_segment_id=100000,
        data_format=DataFormat.Zarr,
    )

    assert (BUCKET_PATH / f"simple_zarr_dataset" / "segmentation").is_dir()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


def test_view_read() -> None:
    delete_dir(BUCKET_PATH / f"simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / f"simple_zarr_dataset",
        BUCKET_PATH / f"simple_zarr_dataset",
    )

    wk_view = (
        Dataset.open(BUCKET_PATH / f"simple_zarr_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel
    assert wk_view.info.data_format == DataFormat.Zarr


def test_view_write() -> None:
    delete_dir(BUCKET_PATH / f"simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / f"simple_zarr_dataset",
        BUCKET_PATH / f"simple_zarr_dataset",
    )

    wk_view = (
        Dataset.open(BUCKET_PATH / f"simple_zarr_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    assert wk_view.info.data_format == DataFormat.Zarr

    np.random.seed(1234)
    write_data = (np.random.rand(3, 10, 10, 10) * 255).astype(np.uint8)

    wk_view.write(write_data)

    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert np.array_equal(data, write_data)


def test_view_write_out_of_bounds() -> None:
    new_dataset_path = BUCKET_PATH / f"zarr_view_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / f"simple_zarr_dataset", new_dataset_path)

    view = (
        Dataset.open(new_dataset_path)
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    with pytest.raises(AssertionError):
        view.write(
            np.zeros((200, 200, 5), dtype=np.uint8)
        )  # this is bigger than the bounding_box


def test_mag_view_write_out_of_bounds() -> None:
    new_dataset_path = BUCKET_PATH / f"simple_zarr_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / f"simple_zarr_dataset", new_dataset_path)

    ds = Dataset.open(new_dataset_path)
    mag_view = ds.get_layer("color").get_mag("1")

    assert mag_view.info.data_format == DataFormat.Zarr

    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 24)
    mag_view.write(
        np.zeros((3, 1, 1, 48), dtype=np.uint8)
    )  # this is bigger than the bounding_box
    assert tuple(ds.get_layer("color").bounding_box.size) == (24, 24, 48)

    assure_exported_properties(ds)


def test_mag_view_write_out_of_bounds_mag2() -> None:
    new_dataset_path = BUCKET_PATH / f"simple_zarr_dataset_out_of_bounds"

    delete_dir(new_dataset_path)
    copytree(TESTDATA_DIR / f"simple_zarr_dataset", new_dataset_path)

    ds = Dataset.open(new_dataset_path)
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


def test_update_new_bounding_box_offset() -> None:
    delete_dir(BUCKET_PATH / "zarr_dataset")

    ds = Dataset(BUCKET_PATH / "zarr_dataset", scale=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY)
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


def test_write_multi_channel_uint8() -> None:
    dataset_path = BUCKET_PATH / f"zarr_multichannel"
    delete_dir(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr
    ).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    mag.write(data)

    assert np.array_equal(data, mag.read())

    assure_exported_properties(ds)


def test_wkw_write_multi_channel_uint16() -> None:
    dataset_path = BUCKET_PATH / f"zarr_multichannel"
    delete_dir(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color",
        COLOR_CATEGORY,
        num_channels=3,
        dtype_per_layer="uint48",
        data_format=DataFormat.Zarr,
    ).add_mag("1")

    data = get_multichanneled_data(np.uint16)

    mag.write(data)
    written_data = mag.read()

    assert np.array_equal(data, written_data)

    assure_exported_properties(ds)

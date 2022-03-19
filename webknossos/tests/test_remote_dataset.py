import multiprocessing
import warnings

import numpy as np
import pytest
import s3fs  # pylint: disable=unused-import
from upath import UPath as Path

from tests.constants import TESTDATA_DIR
from webknossos.dataset import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
    Dataset,
)
from webknossos.geometry import Vec3Int
from webknossos.utils import copytree, rmtree

S3_KEY = "ANTN35UAENTS5UIAEATD"
S3_SECRET = "TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
S3_ENDPOINT = "http://localhost:9000"

BUCKET_PATH = Path(
    "s3://testoutput",
    key=S3_KEY,
    secret=S3_SECRET,
    client_kwargs={"endpoint_url": S3_ENDPOINT},
)

pytestmark = [pytest.mark.block_network(allowed_hosts=[".*"])]

# `s3fs`` hangs in multiprocessing when using `fork`
# See: https://github.com/fsspec/s3fs/issues/464
multiprocessing.set_start_method("forkserver", force=True)


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


@pytest.fixture(scope="session", autouse=True)
def create_bucket() -> None:
    BUCKET_PATH.fs.mkdirs("testoutput", exist_ok=True)


def test_s3_dataset() -> None:
    rmtree(BUCKET_PATH / "zarr_dataset")

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
    rmtree(BUCKET_PATH / "zarr_dataset")

    ds = Dataset(BUCKET_PATH / "zarr_dataset", scale=(1, 1, 1))
    ds.add_layer("color", "color", data_format=DataFormat.Zarr)

    ds.get_layer("color").add_mag("1")
    ds.get_layer("color").add_mag("2-2-1")

    assert (BUCKET_PATH / "zarr_dataset" / "color" / "1-1-1").exists()
    assert (BUCKET_PATH / "zarr_dataset" / "color" / "1-1-1" / ".zarray").exists()
    assert (BUCKET_PATH / "zarr_dataset" / "color" / "2-2-1").exists()
    assert (BUCKET_PATH / "zarr_dataset" / "color" / "2-2-1" / ".zarray").exists()

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 2

    assure_exported_properties(ds)


def test_open_dataset() -> None:
    rmtree(BUCKET_PATH / "simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / "simple_zarr_dataset",
        BUCKET_PATH / "simple_zarr_dataset",
    )
    ds = Dataset.open(BUCKET_PATH / "simple_zarr_dataset")

    assert len(ds.layers) == 1
    assert len(ds.get_layer("color").mags) == 1
    assert ds.get_layer("color").data_format == DataFormat.Zarr


def test_modify_existing_dataset() -> None:
    rmtree(BUCKET_PATH / "simple_zarr_dataset")
    ds1 = Dataset(BUCKET_PATH / "simple_zarr_dataset", scale=(1, 1, 1))
    ds1.add_layer(
        "color",
        COLOR_CATEGORY,
        dtype_per_layer="float",
        num_channels=1,
        data_format=DataFormat.Zarr,
    )

    ds2 = Dataset.open(BUCKET_PATH / "simple_zarr_dataset")

    ds2.add_layer(
        "segmentation",
        SEGMENTATION_CATEGORY,
        "uint8",
        largest_segment_id=100000,
        data_format=DataFormat.Zarr,
    ).add_mag("1")

    assert (
        BUCKET_PATH / "simple_zarr_dataset" / "segmentation" / "1-1-1" / ".zarray"
    ).exists()

    # Note: ds1 is outdated because the same dataset was opened again and changed.
    assure_exported_properties(ds2)


def test_view_read() -> None:
    rmtree(BUCKET_PATH / "simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / "simple_zarr_dataset",
        BUCKET_PATH / "simple_zarr_dataset",
    )

    wk_view = (
        Dataset.open(BUCKET_PATH / "simple_zarr_dataset")
        .get_layer("color")
        .get_mag("1")
        .get_view(absolute_offset=(0, 0, 0), size=(16, 16, 16))
    )

    # 'read()' checks if it was already opened. If not, it opens it automatically
    data = wk_view.read(absolute_offset=(0, 0, 0), size=(10, 10, 10))
    assert data.shape == (3, 10, 10, 10)  # three channel
    assert wk_view.info.data_format == DataFormat.Zarr


def test_view_write() -> None:
    rmtree(BUCKET_PATH / "simple_zarr_dataset")
    copytree(
        TESTDATA_DIR / "simple_zarr_dataset",
        BUCKET_PATH / "simple_zarr_dataset",
    )

    wk_view = (
        Dataset.open(BUCKET_PATH / "simple_zarr_dataset")
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
    new_dataset_path = BUCKET_PATH / "zarr_view_dataset_out_of_bounds"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

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
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_out_of_bounds"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

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
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_out_of_bounds"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

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
    rmtree(BUCKET_PATH / "zarr_dataset")

    ds = Dataset(BUCKET_PATH / "zarr_dataset", scale=(1, 1, 1))
    color_layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.Zarr)
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
    dataset_path = BUCKET_PATH / "zarr_multichannel"
    rmtree(dataset_path)

    ds = Dataset(dataset_path, scale=(1, 1, 1))
    mag = ds.add_layer(
        "color", COLOR_CATEGORY, num_channels=3, data_format=DataFormat.Zarr
    ).add_mag("1")

    data = get_multichanneled_data(np.uint8)

    mag.write(data)

    assert np.array_equal(data, mag.read())

    assure_exported_properties(ds)


def test_write_multi_channel_uint16() -> None:
    dataset_path = BUCKET_PATH / "zarr_multichannel"
    rmtree(dataset_path)

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


@pytest.mark.xfail(raises=AssertionError)
def test_compression() -> None:
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_compression"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

    mag1 = Dataset.open(new_dataset_path).get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100))

    assert not mag1._is_compressed()
    mag1.compress()


def test_compression_with_target_path() -> None:
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_compression"
    compressed_dataset_path = BUCKET_PATH / "simple_zarr_dataset_compressed"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

    mag1 = Dataset.open(new_dataset_path).get_layer("color").get_mag(1)

    # writing unaligned data to an uncompressed dataset
    write_data = (np.random.rand(3, 10, 20, 30) * 255).astype(np.uint8)
    mag1.write(write_data, absolute_offset=(60, 80, 100))

    assert not mag1._is_compressed()

    mag1.compress(
        target_path=compressed_dataset_path,
    )

    mag1 = Dataset.open(compressed_dataset_path).get_layer("color").get_mag(1)
    assert mag1._is_compressed()
    assert mag1.info.data_format == DataFormat.Zarr

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


def test_downsampling() -> None:
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_downsampling"

    rmtree(new_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", new_dataset_path)

    color_layer = Dataset.open(new_dataset_path).get_layer("color")
    color_layer.downsample()

    assert (new_dataset_path / "color" / "2-2-2" / ".zarray").exists()
    assert (new_dataset_path / "color" / "4-4-4" / ".zarray").exists()

    assure_exported_properties(color_layer.dataset)


def test_copy_dataset() -> None:
    new_dataset_path = BUCKET_PATH / "simple_zarr_dataset_copied"

    rmtree(new_dataset_path)

    Dataset.open(TESTDATA_DIR / "simple_zarr_dataset").copy_dataset(
        new_dataset_path,
        chunks_per_shard=1,
        data_format=DataFormat.Zarr,
    )
    assert (new_dataset_path / "color" / "1-1-1" / ".zarray").exists()


def test_add_symlink_layer() -> None:
    src_dataset_path = BUCKET_PATH / "simple_zarr_dataset"
    dst_dataset_path = BUCKET_PATH / "simple_zarr_dataset_symlinks"

    rmtree(src_dataset_path)
    rmtree(dst_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", src_dataset_path)

    src_ds = Dataset.open(src_dataset_path)
    dst_ds = Dataset(dst_dataset_path, scale=(1, 1, 1))

    with pytest.raises(AssertionError):
        dst_ds.add_symlink_layer(src_ds.get_layer("color"))


def test_add_symlink_mag() -> None:
    src_dataset_path = BUCKET_PATH / "simple_zarr_dataset"
    dst_dataset_path = BUCKET_PATH / "simple_zarr_dataset_symlinks"

    rmtree(src_dataset_path)
    rmtree(dst_dataset_path)
    copytree(TESTDATA_DIR / "simple_zarr_dataset", src_dataset_path)

    src_ds = Dataset.open(src_dataset_path)
    src_layer = src_ds.get_layer("color")
    src_mag1 = src_layer.get_mag("1")

    dst_ds = Dataset(dst_dataset_path, scale=(1, 1, 1))
    dst_layer = dst_ds.add_layer("color", COLOR_CATEGORY, dtype_per_channel="uint8")

    with pytest.raises(AssertionError):
        dst_layer.add_symlink_mag(src_mag1)

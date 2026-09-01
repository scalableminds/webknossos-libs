import numpy as np
import pytest
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from tests.dataset._dataset_helpers import (
    DATA_FORMATS,
    DATA_FORMATS_AND_OUTPUT_PATHS,
    copy_simple_dataset,
    prepare_dataset_path,
)
from tests.utils import TestTemporaryDirectoryNonLocal
from webknossos.dataset import (
    Dataset,
    RemoteDataset,
    RemoteFolder,
)
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    DataFormat,
)
from webknossos.geometry import (
    BoundingBox,
)
from webknossos.geometry.constants import (
    C_AXIS,
)
from webknossos.utils import (
    is_fs_path,
)

rng = np.random.default_rng(1234)

pytestmark = pytest.mark.usefixtures("moto_server")


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


@pytest.mark.skip_on_windows
def test_explore_and_add_remote() -> None:
    remote_ds = RemoteDataset.explore_and_add_remote(
        # l4_sample from the test database
        "http://localhost:9000/data/v15/zarr/59e9cfbdba632ac2ab8b23b5/",
        "added_remote_ds",
        folder=RemoteFolder.get_by_path("Organization_X"),
    )
    assert remote_ds.name == "added_remote_ds"


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
        "color", COLOR_CATEGORY, dtype="uint8", data_format=data_format
    )
    assert not dst_layer.read_only

    with pytest.raises(AssertionError):
        dst_layer.add_symlink_mag(src_mag1)


def test_write_remote_wkw_dataset() -> None:
    ds_path = prepare_dataset_path(DataFormat.WKW, REMOTE_TESTOUTPUT_DIR)
    ds = Dataset(ds_path, voxel_size=(1, 1, 1))
    with pytest.warns(UserWarning, match=".*not recommended.*"):
        layer = ds.add_layer("color", COLOR_CATEGORY, data_format=DataFormat.WKW)
    mag = layer.add_mag(1, shard_shape=(256, 256, 256))
    data: np.ndarray = rng.integers(0, 256, (128, 128, 128), dtype=np.uint8)
    mag.write(data, absolute_offset=(0, 0, 0), allow_resize=True)
    actual = mag.read(absolute_bounding_box=BoundingBox((0, 0, 0), (128, 128, 128)))[0]
    np.testing.assert_array_equal(data, actual)


def test_read_remote_wkw_dataset() -> None:
    local_ds_path = copy_simple_dataset(DataFormat.WKW, TESTOUTPUT_DIR, "local")
    remote_ds_path = copy_simple_dataset(
        DataFormat.WKW, REMOTE_TESTOUTPUT_DIR, "remote"
    )
    local_ds = Dataset.open(local_ds_path)
    remote_ds = Dataset.open(remote_ds_path)
    np.testing.assert_equal(
        local_ds.get_layer("color").get_mag("1").read(),
        remote_ds.get_layer("color").get_mag("1").read(),
    )


@pytest.mark.skip_on_windows
def test_remote_dataset_access_metadata() -> None:
    ds = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    assert len(ds.metadata) == 2  # has 2 by default

    ds.metadata["key"] = "value"
    assert ds.metadata["key"] == "value"

    ds.metadata["number"] = 42
    assert ds.metadata["number"] == 42

    ds.metadata["list"] = ["a", "b", C_AXIS]
    assert ds.metadata["list"] == ["a", "b", C_AXIS]

    assert len(ds.folder.metadata) == 1

    ds.folder.metadata["folder_key"] = "folder_value"
    assert ds.folder.metadata["folder_key"] == "folder_value"
    assert len(ds.folder.metadata) == 2


@pytest.mark.skip_on_windows
def test_remote_dataset_urls() -> None:
    ds = RemoteDataset.open("l4_sample", organization_id="Organization_X")
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


def test_create_dataset_remote_storage() -> None:
    """Test creating a dataset with remote storage."""
    with TestTemporaryDirectoryNonLocal() as temp_dir:
        dataset = Dataset(temp_dir / "ds", voxel_size=(10, 10, 10), exist_ok=True)
        layer = dataset.add_layer(
            "color",
            COLOR_CATEGORY,
            data_format="zarr3",
            bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
        )
        mag1 = layer.add_mag(1)
        mag1.write(np.ones((16, 16, 16), dtype="uint8"))
        ds = Dataset.open(temp_dir / "ds")
        read_data = ds.get_layer("color").get_mag(1).read()
        assert read_data.shape == (1, 16, 16, 16)
        assert read_data.dtype == np.uint8
        assert np.all(read_data == 1)

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pytest

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
    use_minio,
)
from webknossos.dataset import Dataset, View
from webknossos.dataset._array import DataFormat
from webknossos.geometry import Mag, Vec3Int
from webknossos.utils import copytree, get_executor_for_args, rmtree


@pytest.fixture(autouse=True, scope="module")
def start_minio() -> Iterator[None]:
    with use_minio():
        yield


DATA_FORMATS = [DataFormat.WKW, DataFormat.Zarr]
DATA_FORMATS_AND_OUTPUT_PATHS = [
    (DataFormat.WKW, TESTOUTPUT_DIR),
    (DataFormat.Zarr, TESTOUTPUT_DIR),
    (DataFormat.Zarr, REMOTE_TESTOUTPUT_DIR),
    (DataFormat.Zarr3, TESTOUTPUT_DIR),
    (DataFormat.Zarr3, REMOTE_TESTOUTPUT_DIR),
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
    data_format: DataFormat, chunk_shape: int = 32
) -> Tuple[Vec3Int, Vec3Int]:
    if data_format == DataFormat.Zarr:
        return (Vec3Int.full(chunk_shape * 8), Vec3Int.full(1))
    else:
        return (Vec3Int.full(chunk_shape), Vec3Int.full(8))


def advanced_chunk_job(args: Tuple[View, int]) -> None:
    view, _i = args

    # write different data for each chunk (depending on the topleft of the chunk)
    data = view.read()
    data = np.ones(data.shape, dtype=np.dtype("uint8")) * (
        sum(view.bounding_box.topleft) % 256
    )
    view.write(data)


def for_each_chunking_with_wrong_chunk_shape(view: View) -> None:
    with get_executor_for_args(None) as executor:
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
    with get_executor_for_args(None) as executor:
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
        assert np.array_equal(
            np.ones(chunk_data.shape, dtype=np.dtype("uint8"))
            * (sum(chunk.bounding_box.topleft) % 256),
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
    assert (
        ds._properties == reopened_ds._properties
    ), "The properties did not match after reopening the dataset. This might indicate that the properties were not exported after they were changed in memory."


@pytest.mark.parametrize("data_format,output_path", DATA_FORMATS_AND_OUTPUT_PATHS)
def test_aligned_downsampling(data_format: DataFormat, output_path: Path) -> None:
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
    chunks_per_shard = None
    if data_format == DataFormat.Zarr3:
        # Writing compressed zarr with large shard shape is slow, compare #issue
        chunks_per_shard = (4, 4, 4)
    test_mag = test_layer.add_mag(
        "1", chunks_per_shard=chunks_per_shard, chunk_shape=32
    )
    test_mag.write(
        absolute_offset=(0, 0, 0),
        # assuming the layer has 3 channels:
        data=(np.random.rand(3, 24, 24, 24) * 255).astype(np.uint8),
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
def test_guided_downsampling(data_format: DataFormat, output_path: Path) -> None:
    ds_path = copy_simple_dataset(data_format, output_path, "guided_downsampling")

    input_dataset = Dataset.open(ds_path)
    input_layer = input_dataset.get_layer("color")

    chunks_per_shard = None
    if data_format == DataFormat.Zarr3:
        # Writing compressed zarr with large shard shape is slow, compare #issue
        chunks_per_shard = (4, 4, 4)

    # Adding additional mags to the input dataset for testing
    input_layer.add_mag("2-2-1", chunks_per_shard=chunks_per_shard, chunk_shape=32)
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
    output_mag = output_layer.add_mag(
        finest_input_mag.mag, chunks_per_shard=chunks_per_shard, chunk_shape=32
    )
    # Copying some data into the output dataset
    input_data = finest_input_mag.read(absolute_offset=(0, 0, 0), size=(24, 24, 24))
    output_mag.write(absolute_offset=(0, 0, 0), data=input_data)
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

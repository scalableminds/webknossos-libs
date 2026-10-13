"""Shared fixtures, constants and job functions for the `test_dataset_*` modules."""

import json

import numpy as np
import pytest
import tensorstore
from cluster_tools import get_executor
from upath import UPath

from tests.constants import (
    REMOTE_TESTOUTPUT_DIR,
    TESTDATA_DIR,
    TESTOUTPUT_DIR,
)
from webknossos.dataset import (
    Dataset,
    View,
)
from webknossos.dataset_properties import (
    DataFormat,
)
from webknossos.geometry import (
    Vec3Int,
)
from webknossos.utils import (
    copytree,
    rmtree,
)

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


def write_zarr3_array(
    path: UPath, data: np.ndarray, dimension_names: list[str]
) -> None:
    array = tensorstore.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": data.shape,
                "data_type": str(data.dtype),
                "dimension_names": dimension_names,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1, 8, 32, 32][-data.ndim :]},
                },
            },
        },
        create=True,
    ).result()
    array.write(data).result()


def write_layers_to_properties(dataset_path: UPath, layers: list[dict]) -> Dataset:
    """Creates (or opens) the dataset at `dataset_path` and replaces its layers
    with the given raw layer properties."""
    Dataset(dataset_path, voxel_size=(11, 11, 25), exist_ok=True)
    properties_path = dataset_path / "datasource-properties.json"
    properties = json.loads(properties_path.read_text())
    properties["dataLayers"] = layers
    properties_path.write_text(json.dumps(properties))
    return Dataset.open(dataset_path)


def generic_dimension_names_layer(
    name: str, channel_index: int, path: str | None = None
) -> dict:
    """Properties of a layer that references one channel of a (t, c, z, y, x)
    array with generic dimension names, like the OME-Zarrs of BigStitcher-Spark.
    The axes are only known from the `axisOrder` and `additionalAxes`."""
    return {
        "name": name,
        "category": "color",
        "dataFormat": "zarr3",
        "elementClass": "uint16",
        "numChannels": 1,
        "boundingBox": {"topLeft": [0, 0, 0], "width": 64, "height": 32, "depth": 8},
        "additionalAxes": [{"name": "t", "bounds": [0, 1], "index": 0}],
        "mags": [
            {
                "mag": [1, 1, 1],
                "path": path if path is not None else f"./{name}/1",
                "axisOrder": {"x": 4, "y": 3, "z": 2, "c": 1},
                "channelIndex": channel_index,
            }
        ],
    }


GENERIC_DIMENSION_NAMES = ["dim_4", "dim_3", "dim_2", "dim_1", "dim_0"]

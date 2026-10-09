import json

import numpy as np
import pytest
import tensorstore
from upath import UPath

from webknossos import Dataset
from webknossos.geometry import BoundingBox

rng = np.random.default_rng(42)

# t, c, z, y, x
SHAPE_5D = (1, 3, 8, 32, 64)


def _write_zarr3_array(
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


def _make_dataset(
    dataset_path: UPath, layers: list[dict], array_path: UPath
) -> Dataset:
    """Creates a dataset whose layers all reference channels of the same array."""
    Dataset(dataset_path, voxel_size=(11, 11, 25))
    properties_path = dataset_path / "datasource-properties.json"
    properties = json.loads(properties_path.read_text())
    for layer in layers:
        for mag in layer["mags"]:
            mag["path"] = str(array_path)
    properties["dataLayers"] = layers
    properties_path.write_text(json.dumps(properties))
    return Dataset.open(dataset_path)


def _make_bigstitcher_dataset(tmp_upath: UPath) -> tuple[Dataset, np.ndarray]:
    """Mimics a BigStitcher-Spark OME-Zarr, whose zarr.json only has generic
    dimension names. The actual axes are only known from the layer's
    `axisOrder` and `additionalAxes`."""
    data = rng.integers(0, 65535, SHAPE_5D, dtype=np.uint16)
    array_path = tmp_upath / "fused.ome.zarr" / "0"
    _write_zarr3_array(array_path, data, ["dim_4", "dim_3", "dim_2", "dim_1", "dim_0"])
    layers = [
        {
            "name": f"channel_{channel_index}",
            "category": "color",
            "dataFormat": "zarr3",
            "elementClass": "uint16",
            "numChannels": 1,
            "boundingBox": {
                "topLeft": [0, 0, 0],
                "width": 64,
                "height": 32,
                "depth": 8,
            },
            "additionalAxes": [{"name": "t", "bounds": [0, 1], "index": 0}],
            "mags": [
                {
                    "mag": [1, 1, 1],
                    "axisOrder": {"x": 4, "y": 3, "z": 2, "c": 1},
                    "channelIndex": channel_index,
                }
            ],
        }
        for channel_index in range(3)
    ]
    return _make_dataset(tmp_upath / "dataset", layers, array_path), data


def test_read_with_axis_order_of_generically_named_array(tmp_upath: UPath) -> None:
    dataset, data = _make_bigstitcher_dataset(tmp_upath)

    for channel_index in range(3):
        mag = dataset.get_layer(f"channel_{channel_index}").get_mag(1)
        assert mag.info.bounding_box.axes == ("t", "c", "z", "y", "x")
        assert np.array_equal(mag.read(), data[:, channel_index : channel_index + 1])
        # Sub-views open their own array
        view = mag.get_view(
            absolute_bounding_box=mag.bounding_box.with_bounds("x", 16, 32)
        )
        view._cached_array = None
        assert np.array_equal(
            view.read(), data[:, channel_index : channel_index + 1, :, :, 16:48]
        )


def test_export_with_axis_order_of_generically_named_array(tmp_upath: UPath) -> None:
    pytest.importorskip("tifffile")
    import tifffile

    dataset, data = _make_bigstitcher_dataset(tmp_upath)

    for channel_index in range(3):
        layer = dataset.get_layer(f"channel_{channel_index}")
        out_path = tmp_upath / f"channel_{channel_index}.ome.tif"
        layer.export.as_ome_tiff(output_path=out_path, downsample=False)
        # tifffile drops the size-1 t and c axes on read-back
        assert np.array_equal(tifffile.imread(str(out_path)), data[0, channel_index])


def test_read_3d_bounding_box_with_channel_index(tmp_upath: UPath) -> None:
    data = rng.integers(0, 255, (3, 64, 32, 8), dtype=np.uint8)
    array_path = tmp_upath / "multi_channel" / "1"
    _write_zarr3_array(array_path, data, ["c", "x", "y", "z"])
    layers = [
        {
            "name": "channel_1",
            "category": "color",
            "dataFormat": "zarr3",
            "elementClass": "uint8",
            "numChannels": 1,
            "boundingBox": {
                "topLeft": [0, 0, 0],
                "width": 64,
                "height": 32,
                "depth": 8,
            },
            "mags": [{"mag": [1, 1, 1], "channelIndex": 1}],
        }
    ]
    dataset = _make_dataset(tmp_upath / "dataset", layers, array_path)
    mag = dataset.get_layer("channel_1").get_mag(1)

    assert np.array_equal(mag.read(), data[1:2])
    assert np.array_equal(
        mag.read(absolute_bounding_box=BoundingBox((8, 0, 0), (16, 32, 8))),
        data[1:2, 8:24],
    )

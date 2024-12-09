import numpy as np
from zarrita import Array

import webknossos as wk


def test_add_mag_from_zarrarray(tmp_path):
    dataset = wk.Dataset(
        tmp_path / "test_add_mag_from_zarrarray", voxel_size=(10, 10, 10)
    )
    layer = dataset.add_layer(
        "color",
        wk.COLOR_CATEGORY,
        data_format="zarr3",
        bounding_box=wk.BoundingBox((0, 0, 0), (16, 16, 16)),
    )
    zarr_mag_path = tmp_path / "zarr_data" / "mag1.zarr"
    zarr_data = np.random.randint(0, 255, (16, 16, 16), dtype="uint8")
    zarr_mag = Array.create(
        store=zarr_mag_path, shape=(16, 16, 16), chunk_shape=(8, 8, 8), dtype="uint8"
    )
    zarr_mag[:] = zarr_data

    layer.add_mag_from_zarrarray("1", zarr_mag_path, extend_layer_bounding_box=False)

    assert layer.get_mag("1").read().shape == (1, 16, 16, 16)
    assert layer.get_mag("1").info.num_channels == 1
    assert layer.get_mag("1").info.dimension_names == ("c", "x", "y", "z")
    assert (layer.get_mag("1").read()[0] == zarr_data).all()

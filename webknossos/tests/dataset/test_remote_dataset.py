
import numpy as np

import webknossos as wk
from tests.utils import TestTemporaryDirectoryNonLocal


def test_create_dataset_remote_storage()->None:
    """Test creating a dataset with remote storage."""
    # Create a temporary directory for the dataset
    # with tempfile.TemporaryDirectory() as temp_dir:
    with TestTemporaryDirectoryNonLocal() as temp_dir:
        dataset = wk.Dataset(temp_dir / "ds", voxel_size=(10, 10, 10), exist_ok=True)
        layer = dataset.add_layer(
            "color",
            wk.COLOR_CATEGORY,
            data_format="zarr3",
            bounding_box=wk.BoundingBox((0, 0, 0), (16, 16, 16)),
        )
        mag1 = layer.add_mag(1)
        mag1.write(np.ones((16, 16, 16), dtype="uint8"))
        ds = wk.Dataset.open(temp_dir / "ds")
        read_data = ds.get_layer("color").get_mag(1).read()
        assert read_data.shape == (1, 16, 16, 16)
        assert read_data.dtype == np.uint8
        assert np.all(read_data == 1)

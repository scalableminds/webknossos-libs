import itertools
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest

from webknossos import COLOR_CATEGORY, Dataset, RemoteDataset
from webknossos.geometry import BoundingBox
from webknossos.utils import is_remote_path

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="too slow on windows"),
    pytest.mark.use_proxay,
]


@pytest.fixture
def sample_remote_dataset(tmp_path: Path) -> Iterator[Dataset]:
    yield RemoteDataset.open("l4_sample").download(
        path=tmp_path / "l4_sample",
        bbox=BoundingBox((3457, 3323, 1204), (10, 10, 10)),
    )


@pytest.fixture(scope="module")
def sample_layer_and_mag_name() -> list[tuple[str, str]]:
    layer_names = ["color", "segmentation"]
    mag_names = ["1", "2-2-1", "4-4-1"]
    return list(itertools.product(layer_names, mag_names))


def test_add_remote_mags_from_mag_view(
    sample_remote_dataset: Dataset,
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", "Organization_X")
    sample_remote_mags = [
        remote_dataset.get_layer(layer).get_mag(mag)
        for layer, mag in sample_layer_and_mag_name
    ]
    for remote_mag in sample_remote_mags:
        mag_path = remote_mag.path
        layer_type = remote_mag.layer.category
        assert is_remote_path(mag_path), "Remote mag does not have remote path."
        layer_name = f"test_remote_layer_{mag_path.parent.name}_{mag_path.name}_object"
        new_layer = sample_remote_dataset.add_layer(
            layer_name,
            layer_type,
            data_format=remote_mag.info.data_format,
            dtype_per_channel=remote_mag.get_dtype(),
        )
        new_layer.add_mag_as_ref(remote_mag)
        added_mag = sample_remote_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_mags_from_path(
    sample_remote_dataset: Dataset,
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", "Organization_X")
    sample_remote_mags = [
        remote_dataset.get_layer(layer).get_mag(mag)
        for layer, mag in sample_layer_and_mag_name
    ]
    print(sample_remote_mags)
    for remote_mag in sample_remote_mags:
        mag_path = remote_mag.path
        layer_type = remote_mag.layer.category
        assert is_remote_path(mag_path), "Remote mag does not have remote path."
        # Additional .parent calls are needed as the first .parent only removes the trailing slash.
        layer_name = f"test_remote_layer_{mag_path.parent.name}_{mag_path.name}_path"
        new_layer = sample_remote_dataset.add_layer(
            layer_name,
            layer_type,
            data_format=remote_mag.info.data_format,
            dtype_per_channel=remote_mag.get_dtype(),
        )
        new_layer.add_mag_as_ref(str(remote_mag.path))
        added_mag = sample_remote_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_ref_layer_from_remote_layer(sample_remote_dataset: Dataset) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", "Organization_X")
    assert remote_dataset.zarr_streaming_path is not None, (
        "Zarr streaming sets a remote path."
    )
    assert is_remote_path(remote_dataset.zarr_streaming_path), (
        "zarr streaming path should be remote."
    )
    sample_remote_layer = list(remote_dataset.layers.values())
    for layer in sample_remote_layer:
        layer_name = f"test_remote_layer_{layer.category}_object"
        sample_remote_dataset.add_layer_as_ref(layer, layer_name)
        new_layer = sample_remote_dataset.layers[layer_name]
        assert is_remote_path(new_layer.get_mag(1).path) and (
            str(layer.get_mag(1).path) == str(new_layer.get_mag(1).path)
        ), "Mag path of added layer should be equal to mag path in source layer."


def test_ref_layer_non_public(tmp_path: Path) -> None:
    dataset = Dataset.open("testdata/simple_zarr3_dataset").copy_dataset(
        tmp_path / "simple_zarr3_dataset"
    )
    remote_dataset = RemoteDataset.open("l4_sample", "Organization_X")
    remote_dataset.is_public = False
    dataset.add_layer_as_ref(remote_dataset.get_layer("segmentation"), "segmentation")

    assert dataset.layers["segmentation"].get_mag("16-16-4").read().shape == (
        1,
        64,
        64,
        256,
    )
    remote_dataset.is_public = True


def test_shallow_copy_remote_layers(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "origin", voxel_size=(10, 10, 10))
    remote_dataset = RemoteDataset.open("l4_sample", "Organization_X")
    dataset.add_layer_as_ref(remote_dataset.get_layer("color"), "color")
    dataset_copy = dataset.shallow_copy_dataset(tmp_path / "copy")
    data = dataset_copy.get_layer("color").get_mag("16-16-4").read()
    assert data.shape == (1, 64, 64, 256)


def test_add_mag_ref_from_local_path(tmp_path: Path) -> None:
    dataset1 = Dataset(tmp_path / "origin", voxel_size=(10, 10, 10))
    dataset1.write_layer(
        "color",
        COLOR_CATEGORY,
        data=np.ones((1, 10, 10, 10), dtype="uint8"),
        downsample=False,
    )

    dataset2 = Dataset(tmp_path / "copy", voxel_size=(10, 10, 10))
    layer1 = dataset2.add_layer_as_ref(tmp_path / "origin" / "color")
    layer1_mag1 = layer1.get_mag(1)

    assert layer1_mag1.path == tmp_path / "origin" / "color" / "1"
    assert layer1_mag1._properties.path == str(
        (tmp_path / "origin" / "color" / "1").resolve()
    )

    layer2_mag1 = dataset2.add_layer("color2", COLOR_CATEGORY).add_mag_as_ref(
        tmp_path / "origin" / "color" / "1"
    )
    assert layer2_mag1.path == tmp_path / "origin" / "color" / "1"
    assert layer2_mag1._properties.path == str(
        (tmp_path / "origin" / "color" / "1").resolve()
    )

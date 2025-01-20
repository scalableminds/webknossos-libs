import itertools
import os
from pathlib import Path
from typing import Iterable

import pytest
from upath import UPath

import webknossos as wk
from webknossos.utils import is_remote_path


def get_sample_remote_dataset() -> wk.Dataset:
    return wk.Dataset.download(
        "l4_sample",
        path="testoutput/l4_sample",
        bbox=wk.BoundingBox((3457, 3323, 1204), (10, 10, 10)),
    )


pytestmark = [pytest.mark.use_proxay]


@pytest.fixture(scope="module")
def sample_layer_and_mag_name() -> Iterable[tuple[str, str]]:
    layer_names = ["color", "segmentation"]
    mag_names = ["1", "2-2-1", "4-4-1"]
    return itertools.product(layer_names, mag_names)


def test_add_remote_mags_from_mag_view(
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
) -> None:
    sample_remote_dataset = get_sample_remote_dataset()
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
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
        new_layer.add_remote_mag(remote_mag)
        added_mag = sample_remote_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_mags_from_path(
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
) -> None:
    sample_remote_dataset = get_sample_remote_dataset()
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
    sample_remote_mags = [
        remote_dataset.get_layer(layer).get_mag(mag)
        for layer, mag in sample_layer_and_mag_name
    ]
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
        new_layer.add_remote_mag(str(remote_mag.path))
        added_mag = sample_remote_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_layer_from_object() -> None:
    sample_remote_dataset = get_sample_remote_dataset()
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
    sample_remote_layer = list(remote_dataset.layers.values())
    for layer in sample_remote_layer:
        assert is_remote_path(layer.path), "Remote mag does not have remote path."
        layer_name = f"test_remote_layer_{layer.category}_object"
        sample_remote_dataset.add_remote_layer(layer, layer_name)
        new_layer = sample_remote_dataset.layers[layer_name]
        assert is_remote_path(new_layer.path) and (
            layer.path.as_uri() == new_layer.path.as_uri()
        ), "Added layer should have a remote path matching the remote layer added."


def test_add_remote_layer_from_path() -> None:
    sample_remote_dataset = get_sample_remote_dataset()
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
    sample_remote_layer = list(remote_dataset.layers.values())
    for layer in sample_remote_layer:
        assert is_remote_path(layer.path), "Remote mag does not have remote path."
        layer_name = f"test_remote_layer_{layer.category}_path"
        sample_remote_dataset.add_remote_layer(UPath(layer.path), layer_name)
        new_layer = sample_remote_dataset.layers[layer_name]
        assert is_remote_path(new_layer.path) and (
            new_layer.path.as_uri() == layer.path.as_uri()
        ), "Added layer should have a remote path matching the remote layer added."


def test_add_remote_layer_non_public(tmp_path: Path) -> None:
    dataset = wk.Dataset.open("testdata/simple_zarr3_dataset").copy_dataset(
        tmp_path / "simple_zarr3_dataset"
    )
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
    remote_dataset.is_public = False
    dataset.add_remote_layer(remote_dataset.get_layer("segmentation"), "segmentation")

    assert dataset.layers["segmentation"].get_mag("16-16-4").read().shape == (
        1,
        64,
        64,
        256,
    )
    remote_dataset.is_public = True


def test_shallow_copy_remote_layers(tmp_path: Path) -> None:
    dataset = wk.Dataset(tmp_path / "origin", voxel_size=(10, 10, 10))
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", os.getenv("WK_TOKEN")
    )
    dataset.add_remote_layer(remote_dataset.get_layer("color"), "color")
    dataset_copy = dataset.shallow_copy_dataset(tmp_path / "copy")
    data = dataset_copy.get_layer("color").get_mag("16-16-4").read()
    assert data.shape == (1, 64, 64, 256)

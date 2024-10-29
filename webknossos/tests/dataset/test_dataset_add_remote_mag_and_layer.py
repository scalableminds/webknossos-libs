import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest
from upath import UPath

import webknossos as wk
from webknossos import Dataset, MagView
from webknossos.utils import is_remote_path


@pytest.fixture(scope="module")
def sample_bbox() -> wk.BoundingBox:
    return wk.BoundingBox((2807, 4352, 1794), (10, 10, 10))


@pytest.fixture(scope="module")
def sample_remote_dataset(sample_bbox: wk.BoundingBox) -> Iterator[wk.Dataset]:
    url = "http://localhost:9000/datasets/Organization_X/l4_sample"
    with TemporaryDirectory() as temp_dir:
        yield wk.Dataset.download(url, path=Path(temp_dir) / "ds", bbox=sample_bbox)


pytestmark = [pytest.mark.use_proxay]


@pytest.fixture(scope="module")
def sample_remote_mags() -> list[wk.MagView]:
    mag_urls = [
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/color/1/",
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/color/2-2-1/",
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/color/4-4-2/",
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/segmentation/1/",
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/segmentation/2-2-1/",
        "http://localhost:9000/data/zarr/Organization_X/l4_sample/segmentation/4-4-2/",
    ]
    mags = [MagView._ensure_mag_view(url) for url in mag_urls]
    return mags


@pytest.fixture(scope="module")
def sample_remote_layer() -> list[wk.Layer]:
    token = os.getenv("WK_TOKEN")
    if not token:
        raise EnvironmentError("WK_TOKEN environment variable not set")

    remote_dataset = Dataset.open_remote(
        "l4_sample", "Organization_X", token, "http://localhost:9000"
    )
    return list(remote_dataset.layers.values())


def test_add_remote_mags_from_mag_view(
    sample_remote_mags: list[wk.MagView], sample_remote_dataset: wk.Dataset
) -> None:
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
            added_mag.path == mag_path or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_mags_from_path(
    sample_remote_mags: list[wk.MagView],
    sample_remote_dataset: wk.Dataset,
) -> None:
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
            added_mag.path == mag_path or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_layer_from_object(
    sample_remote_layer: list[wk.Layer], sample_remote_dataset: wk.Dataset
) -> None:
    for layer in sample_remote_layer:
        assert is_remote_path(layer.path), "Remote mag does not have remote path."
        layer_name = f"test_remote_layer_{layer.category}_object"
        sample_remote_dataset.add_remote_layer(layer, layer_name)
        new_layer = sample_remote_dataset.layers[layer_name]
        assert is_remote_path(new_layer.path) and (
            layer.path.as_uri() == new_layer.path.as_uri()
        ), "Added layer should have a remote path matching the remote layer added."


def test_add_remote_layer_from_path(
    sample_remote_layer: list[wk.Layer],
    sample_remote_dataset: wk.Dataset,
) -> None:
    for layer in sample_remote_layer:
        assert is_remote_path(layer.path), "Remote mag does not have remote path."
        layer_name = f"test_remote_layer_{layer.category}_path"
        sample_remote_dataset.add_remote_layer(UPath(layer.path), layer_name)
        new_layer = sample_remote_dataset.layers[layer_name]
        assert is_remote_path(new_layer.path) and (
            new_layer.path.as_uri() == layer.path.as_uri()
        ), "Added layer should have a remote path matching the remote layer added."

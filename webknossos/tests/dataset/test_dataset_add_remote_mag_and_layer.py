import itertools
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Iterable, Iterator, List

import pytest
from upath import UPath

import webknossos as wk
from webknossos.utils import is_remote_path


@pytest.fixture(scope="module")
def sample_remote_dataset() -> Iterator[wk.Dataset]:
    with TemporaryDirectory() as tmpdir:
        original_ds = wk.Dataset.open("testdata/l4_sample_snipped")
        yield original_ds.copy_dataset(tmpdir)


pytestmark = [pytest.mark.use_proxay]


@pytest.fixture(scope="module")
def sample_layer_and_mag_name() -> Iterable[tuple[str, str]]:
    layer_names = ["color", "segmentation"]
    mag_names = ["1", "2-2-1", "4-4-1"]
    return itertools.product(layer_names, mag_names)


@pytest.fixture(scope="module")
def sample_remote_layer() -> list[wk.Layer]:
    os.environ["HTTP_PROXY"] = "http://localhost:3000"
    token = os.getenv("WK_TOKEN")
    remote_dataset = wk.Dataset.open_remote(
        "l4_sample", "Organization_X", token, "http://localhost:9000"
    )
    return list(remote_dataset.layers.values())


def test_add_remote_mags_from_mag_view(
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
    sample_remote_dataset: wk.Dataset,
) -> None:
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
    sample_remote_dataset: wk.Dataset,
) -> None:
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

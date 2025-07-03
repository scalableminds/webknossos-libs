import pickle
from pathlib import Path

import numpy as np
import pytest

import webknossos as wk

SAMPLE_BBOX = wk.BoundingBox((3164, 3212, 1017), (10, 10, 10))

pytestmark = [pytest.mark.use_proxay]


def get_sample_dataset(tmpdir: Path) -> wk.Dataset:
    url = "http://localhost:9000/datasets/Organization_X/l4_sample"
    return wk.Dataset.download(url, path=Path(tmpdir) / "sample_ds", bbox=SAMPLE_BBOX)


def test_get_remote_datasets() -> None:
    datasets = wk.Dataset.get_remote_datasets()
    assert any(ds.name == "l4_sample" for ds in datasets.values())
    l4_sample = wk.Dataset.open_remote("l4_sample")
    l4_sample_id = l4_sample._dataset_id
    assert l4_sample_id in datasets
    l4_from_datasets = datasets[l4_sample_id]
    assert l4_from_datasets.name == l4_sample.name
    assert l4_from_datasets.description == l4_sample.description
    assert l4_from_datasets.tags == l4_sample.tags
    assert l4_from_datasets.folder == l4_sample.folder

    datasets_by_name = wk.Dataset.get_remote_datasets(name="l4_sample")
    assert len(datasets_by_name) == 1

    datasets_by_organization = wk.Dataset.get_remote_datasets(
        organization_id="Organization_X"
    )
    assert len(datasets_by_organization) > 0


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000/datasets/Organization_X/l4_sample",
        "http://localhost:9000/datasets/Organization_X/l4_sample/view",
        # "http://localhost:9000/datasets/scalable_minds/l4_sample_dev_sharing/view?token=ilDXmfQa2G8e719vb1U9YQ#%7B%22orthogonal%7D",
        # "http://localhost:9000/links/93zLg9U9vJ3c_UWp",
    ],
)
def test_url_download(url: str, tmp_path: Path) -> None:
    sample_dataset = get_sample_dataset(tmp_path)
    ds = wk.Dataset.download(
        url, path=tmp_path / "ds", mags=[wk.Mag(1)], bbox=SAMPLE_BBOX
    )
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = ds.get_color_layers()[0].get_finest_mag().read()
    assert data.sum() == 120697
    assert np.array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9000/datasets/Organization_X/l4_sample",
        "http://localhost:9000/datasets/Organization_X/l4_sample/view",
        # "http://localhost:9000/datasets/Organization_X/l4_sample_dev_sharing/view?token=ilDXmfQa2G8e719vb1U9YQ#%7B%22orthogonal%7D",
        # "http://localhost:9000/links/93zLg9U9vJ3c_UWp",
    ],
)
def test_url_open_remote(url: str, tmp_path: Path) -> None:
    sample_dataset = get_sample_dataset(tmp_path)
    ds = wk.Dataset.open_remote(url)
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = (
        ds.get_color_layers()[0]
        .get_finest_mag()
        .read(absolute_bounding_box=SAMPLE_BBOX)
    )
    assert data.sum() == 120697
    assert np.array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )
    assert set(pickle.loads(pickle.dumps(ds)).layers.keys()) == {
        "color",
        "segmentation",
    }, "Dataset instances should be picklable."


def test_remote_dataset(tmp_path: Path) -> None:
    sample_dataset = get_sample_dataset(tmp_path)
    remote_ds = sample_dataset.upload(new_dataset_name="test_remote_metadata")
    assert np.array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )

    assert remote_ds.read_only
    assert remote_ds.get_color_layers()[0].read_only
    assert remote_ds.get_color_layers()[0].get_finest_mag().read_only

    assert remote_ds.name == "test_remote_metadata"

    assert remote_ds.description is None
    remote_ds.description = "My awesome test description"
    assert remote_ds.description == "My awesome test description"
    del remote_ds.description
    assert remote_ds.description is None

    assert not remote_ds.is_public
    remote_ds.is_public = True
    assert remote_ds.is_public

    assert len(remote_ds.tags) == 0
    for i in range(3):
        remote_ds.tags += (f"category {i}",)
    assert remote_ds.tags == ("category 0", "category 1", "category 2")

    assert len(remote_ds.sharing_token) > 0

    assert len(remote_ds.allowed_teams) == 0
    test_teams = (wk.Team.get_by_name("team_X1"),)
    assert test_teams[0].id == "570b9f4b2a7c0e3b008da6ec"
    remote_ds.allowed_teams = test_teams
    assert remote_ds.allowed_teams == test_teams
    remote_ds.allowed_teams = ["570b9f4b2a7c0e3b008da6ec"]
    assert remote_ds.allowed_teams == test_teams
    remote_ds.folder = wk.RemoteFolder.get_by_path("Organization_X/A subfolder!")
    assert remote_ds.folder.name == "A subfolder!"


def test_upload_download_roundtrip(tmp_path: Path) -> None:
    ds_original = get_sample_dataset(tmp_path)
    uploaded_dataset = ds_original.upload(
        new_dataset_name="test_upload_download_roundtrip"
    )
    wk.Dataset.trigger_reload_in_datastore(
        "test_upload_download_roundtrip", "Organization_X"
    )
    ds_roundtrip = wk.Dataset.download(
        uploaded_dataset.url, path=tmp_path / "ds", layers=["color", "segmentation"]
    )
    assert set(ds_original.get_segmentation_layers()[0].mags.keys()) == set(
        ds_roundtrip.get_segmentation_layers()[0].mags.keys()
    )

    original_config = ds_original.get_layer("color").default_view_configuration
    roundtrip_config = ds_roundtrip.get_layer("color").default_view_configuration
    assert original_config is not None, (
        "default_view_configuration should be defined for original dataset"
    )
    assert roundtrip_config is not None, (
        "default_view_configuration should be defined for roundtrip dataset"
    )
    assert original_config.color == roundtrip_config.color
    assert original_config.intensity_range == roundtrip_config.intensity_range

    data_original = ds_original.get_segmentation_layers()[0].get_finest_mag().read()
    data_roundtrip = ds_roundtrip.get_segmentation_layers()[0].get_finest_mag().read()
    assert np.array_equal(data_original, data_roundtrip)


def test_upload_twice(tmp_path: Path) -> None:
    ds_original = get_sample_dataset(tmp_path)
    remote1 = ds_original.upload(new_dataset_name="test_upload_twice")
    remote2 = ds_original.upload(new_dataset_name="test_upload_twice")
    assert remote1.url != remote2.url
    assert remote1.name == remote2.name

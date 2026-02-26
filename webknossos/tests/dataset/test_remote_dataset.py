import inspect
import itertools
import pickle
import sys
from collections.abc import Iterable, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from upath import UPath

from tests.constants import TESTOUTPUT_DIR
from webknossos import (
    COLOR_CATEGORY,
    BoundingBox,
    DataFormat,
    Dataset,
    LayerViewConfiguration,
    Mag,
    RemoteAccessMode,
    RemoteDataset,
    RemoteFolder,
    Team,
    TransferMode,
)
from webknossos.utils import is_remote_path, rmtree

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="too slow on windows"),
    pytest.mark.use_proxay,
]

SAMPLE_BBOX = BoundingBox((3164, 3212, 1017), (10, 10, 10))


@pytest.fixture
def sample_remote_dataset(tmp_upath: UPath) -> Iterator[Dataset]:
    yield RemoteDataset.open("l4_sample").download(
        path=tmp_upath / "l4_sample",
        bbox=BoundingBox((3457, 3323, 1204), (10, 10, 10)),
    )


@pytest.fixture(scope="module")
def sample_layer_and_mag_name() -> list[tuple[str, str]]:
    layer_names = ["color", "segmentation"]
    mag_names = ["1", "2-2-1", "4-4-1"]
    return list(itertools.product(layer_names, mag_names))


def _prepare_dataset_path(output_path: UPath, suffix: str) -> UPath:
    new_dataset_path = output_path / f"zarr3_dataset_{suffix}"
    rmtree(new_dataset_path)
    return new_dataset_path


def test_remote_dataset_add_layer_as_copy() -> None:
    """Test that RemoteDataset.add_layer_as_copy correctly delegates to add_layer and add_mag_as_copy."""
    ds_path = _prepare_dataset_path(TESTOUTPUT_DIR, "remote_copy_src")
    source_ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    source_layer = source_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=DataFormat.Zarr3
    )
    source_layer.add_mag(1).write(
        absolute_offset=(0, 0, 0),
        data=(np.random.rand(16, 16, 16) * 255).astype(np.uint8),
        allow_resize=True,
    )

    remote_ds = source_ds.upload(
        new_dataset_name="test_remote_dataset_add_layer_as_copy"
    )
    remote_ds = RemoteDataset.open(
        dataset_id=remote_ds.dataset_id, access_mode=RemoteAccessMode.DIRECT_PATH
    )
    remote_ds.add_layer_as_copy(
        source_layer, new_layer_name="color2", transfer_mode=TransferMode.COPY
    )
    assert len(remote_ds.layers) == 2


def test_add_remote_mags_from_mag_view(
    sample_remote_dataset: Dataset,
    sample_layer_and_mag_name: Iterable[tuple[str, str]],
) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
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
            dtype=remote_mag.get_dtype(),
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
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
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
            dtype=remote_mag.get_dtype(),
        )
        new_layer.add_mag_as_ref(str(remote_mag.path))
        added_mag = sample_remote_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_ref_layer_from_remote_layer(sample_remote_dataset: Dataset) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
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


def test_ref_layer_non_public(tmp_upath: UPath) -> None:
    dataset = Dataset.open("testdata/simple_zarr3_dataset").copy_dataset(
        tmp_upath / "simple_zarr3_dataset"
    )
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    remote_dataset.is_public = False
    dataset.add_layer_as_ref(remote_dataset.get_layer("segmentation"), "segmentation")

    assert dataset.layers["segmentation"].get_mag("16-16-4").read().shape == (
        1,
        64,
        64,
        256,
    )
    remote_dataset.is_public = True


def test_shallow_copy_remote_layers(tmp_upath: UPath) -> None:
    dataset = Dataset(tmp_upath / "origin", voxel_size=(10, 10, 10))
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    dataset.add_layer_as_ref(remote_dataset.get_layer("color"), "color")
    dataset_copy = dataset.shallow_copy_dataset(tmp_upath / "copy")
    data = dataset_copy.get_layer("color").get_mag("16-16-4").read()
    assert data.shape == (1, 64, 64, 256)


def test_add_mag_ref_from_local_path(tmp_upath: UPath) -> None:
    dataset1 = Dataset(tmp_upath / "origin", voxel_size=(10, 10, 10))
    dataset1.write_layer(
        "color",
        COLOR_CATEGORY,
        data=np.ones((1, 10, 10, 10), dtype="uint8"),
        downsample=False,
    )

    dataset2 = Dataset(tmp_upath / "copy", voxel_size=(10, 10, 10))
    layer1 = dataset2.add_layer_as_ref(tmp_upath / "origin" / "color")
    layer1_mag1 = layer1.get_mag(1)

    assert layer1_mag1.path == tmp_upath / "origin" / "color" / "1"
    assert layer1_mag1._properties.path == str(
        (tmp_upath / "origin" / "color" / "1").resolve()
    )

    layer2_mag1 = dataset2.add_layer("color2", COLOR_CATEGORY).add_mag_as_ref(
        tmp_upath / "origin" / "color" / "1"
    )
    assert layer2_mag1.path == tmp_upath / "origin" / "color" / "1"
    assert layer2_mag1._properties.path == str(
        (tmp_upath / "origin" / "color" / "1").resolve()
    )


def test_changing_properties_on_remote_dataset() -> None:
    remote_dataset = RemoteDataset.open(dataset_id="59e9cfbdba632ac2ab8b23b5")
    remote_dataset.description = "This is a test description"
    assert remote_dataset.description == "This is a test description"
    largest_segment_id_before_change_attempt = remote_dataset.get_segmentation_layer(
        "segmentation"
    ).largest_segment_id
    with pytest.raises(RuntimeError):
        remote_dataset.get_segmentation_layer("segmentation").largest_segment_id = 10
    assert (
        remote_dataset.get_segmentation_layer("segmentation").largest_segment_id
        == largest_segment_id_before_change_attempt
    )
    default_view_configuration_before_change_attempt = remote_dataset.get_layer(
        "color"
    ).default_view_configuration
    with pytest.raises(RuntimeError):
        remote_dataset.get_layer(
            "color"
        ).default_view_configuration = LayerViewConfiguration(alpha=0.3)
    assert (
        remote_dataset.get_layer("color").default_view_configuration
        == default_view_configuration_before_change_attempt
    )


def test_changing_properties_on_read_only_remote_dataset() -> None:
    remote_dataset = RemoteDataset.open(
        dataset_id="59e9cfbdba632ac2ab8b23b5", read_only=True
    )
    description_before_change_attempt = remote_dataset.description
    with pytest.raises(RuntimeError):
        remote_dataset.description = "This is a test description"
    assert remote_dataset.description == description_before_change_attempt


def get_sample_dataset(tmpdir: UPath) -> Dataset:
    url = "http://localhost:9000/datasets/Organization_X/l4_sample"
    return RemoteDataset.open(url).download(
        path=UPath(tmpdir) / "sample_ds", bbox=SAMPLE_BBOX
    )


def test_get_remote_datasets() -> None:
    datasets = RemoteDataset.list()
    assert any(ds.name == "l4_sample" for ds in datasets.values())
    l4_sample = RemoteDataset.open("l4_sample")
    l4_sample_id = l4_sample._dataset_id
    assert l4_sample_id in datasets
    l4_from_datasets = datasets[l4_sample_id]
    assert l4_from_datasets.name == l4_sample.name
    assert l4_from_datasets.description == l4_sample.description
    assert l4_from_datasets.tags == l4_sample.tags
    assert l4_from_datasets.folder == l4_sample.folder

    datasets_by_name = RemoteDataset.list(name="l4_sample")
    assert len(datasets_by_name) == 1

    datasets_by_organization = RemoteDataset.list(organization_id="Organization_X")
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
def test_url_download(url: str, tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    ds = RemoteDataset.open(url).download(
        path=tmp_upath / "ds", mags=[Mag(1)], bbox=SAMPLE_BBOX
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
@pytest.mark.parametrize(
    "access_mode", [RemoteAccessMode.ZARR_STREAMING, RemoteAccessMode.PROXY_PATH]
)
def test_url_open_remote(
    url: str, tmp_upath: UPath, access_mode: RemoteAccessMode
) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    ds = RemoteDataset.open(url, access_mode=access_mode)
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


@pytest.mark.skip(
    reason="This won't work in CI as the paths stored in cassettes are always absolute and dependent on the system recording the cassette."
)
def test_upload_dataset_with_symlinks(tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    remote_ds = sample_dataset.upload(
        new_dataset_name="test_remote_symlink",
        transfer_mode=TransferMode.MOVE_AND_SYMLINK,
    )
    assert np.array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


@pytest.mark.skip(
    reason="This could work in CI with a local minio instance. Configuring webknossos is a bit more involved and, therefore, future work."
)
def test_upload_dataset_copy_to_paths(tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    remote_ds = sample_dataset.upload(
        new_dataset_name="test_remote_copy", transfer_mode=TransferMode.COPY
    )
    assert np.array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )


def test_remote_dataset(tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    remote_ds = sample_dataset.upload(new_dataset_name="test_remote_metadata")
    assert np.array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )

    assert not remote_ds.read_only
    assert remote_ds.get_color_layers()[0].read_only
    assert remote_ds.get_color_layers()[0].get_finest_mag().read_only

    assert remote_ds.name == "test_remote_metadata"
    assert remote_ds.used_storage_bytes == 1188727

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
    test_teams = (Team.get_by_name("team_X1"),)
    assert test_teams[0].id == "570b9f4b2a7c0e3b008da6ec"
    remote_ds.allowed_teams = test_teams
    assert remote_ds.allowed_teams == test_teams
    remote_ds.allowed_teams = ["570b9f4b2a7c0e3b008da6ec"]
    assert remote_ds.allowed_teams == test_teams
    remote_ds.folder = RemoteFolder.get_by_path("Organization_X/A subfolder!")
    assert remote_ds.folder.name == "A subfolder!"


def test_folders_and_teams() -> None:
    folder_name = "test_folder"
    team_name = "test_team"

    remote_folder = RemoteFolder.get_root().add_subfolder(folder_name)
    assert remote_folder.name == folder_name

    remote_team = Team.add(team_name)
    remote_folder.allowed_teams = (remote_team,)
    assert remote_folder.allowed_teams == (remote_team,)

    remote_folder.name = f"{folder_name}_renamed"
    assert remote_folder.name == f"{folder_name}_renamed"

    subfolder = remote_folder.add_subfolder(f"{folder_name}_subfolder")
    assert remote_folder.get_subfolders() == (subfolder,)
    subfolder.parent = RemoteFolder.get_root()
    assert remote_folder.get_subfolders() == ()
    subfolder.delete()
    assert remote_folder.get_subfolders() == ()

    remote_folder.delete()
    remote_team.delete()


def test_upload_download_roundtrip(tmp_upath: UPath) -> None:
    ds_original = get_sample_dataset(tmp_upath)
    uploaded_dataset = ds_original.upload(
        new_dataset_name="test_upload_download_roundtrip"
    )
    RemoteDataset.trigger_reload_in_datastore(
        dataset_name_or_url="test_upload_download_roundtrip",
        organization_id="Organization_X",
    )
    ds_roundtrip = RemoteDataset.open(uploaded_dataset.url).download(
        path=tmp_upath / "ds", layers=["color", "segmentation"]
    )
    assert set(ds_original.get_segmentation_layers()[0].mags.keys()) == set(
        ds_roundtrip.get_segmentation_layers()[0].mags.keys()
    )

    data_original = ds_original.get_segmentation_layers()[0].get_finest_mag().read()
    data_roundtrip = ds_roundtrip.get_segmentation_layers()[0].get_finest_mag().read()
    assert np.array_equal(data_original, data_roundtrip)


def test_upload_twice(tmp_upath: UPath) -> None:
    ds_original = get_sample_dataset(tmp_upath)
    remote1 = ds_original.upload(new_dataset_name="test_upload_twice")
    remote2 = ds_original.upload(new_dataset_name="test_upload_twice")
    assert remote1.url != remote2.url
    assert remote1.name == remote2.name

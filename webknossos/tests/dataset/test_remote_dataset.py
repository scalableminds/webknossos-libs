import itertools
import pickle
from collections.abc import Iterable, Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from upath import UPath

from tests.constants import TESTDATA_DIR, TESTOUTPUT_DIR
from webknossos import (
    COLOR_CATEGORY,
    AffineCoordinateTransformation,
    AgglomerateAttachment,
    AgglomerateGraph,
    Annotation,
    BoundingBox,
    DataFormat,
    Dataset,
    LayerViewConfiguration,
    Mag,
    RemoteAccessMode,
    RemoteDataset,
    RemoteFolder,
    SegmentationLayer,
    Team,
    ThinPlateSplineCoordinateTransformation,
    TransferMode,
    Vec3Int,
    webknossos_context,
)
from webknossos.dataset.layer.remote_layer import RemoteLayer
from webknossos.dataset.layer.segmentation_layer.attachments.attachments import (
    RemoteAttachments,
)
from webknossos.utils import is_remote_path, rmtree

pytestmark = [pytest.mark.skip_on_windows]

SAMPLE_BBOX = BoundingBox((3164, 3212, 1017), (10, 10, 10))


@pytest.fixture
def sample_downloaded_dataset(tmp_upath: UPath) -> Iterator[Dataset]:
    yield RemoteDataset.open("l4_sample").download(
        path=tmp_upath / "l4_sample",
        bounding_box=BoundingBox((3457, 3323, 1204), (10, 10, 10)),
    )


def get_sample_dataset(
    tmpdir: UPath, *, layers: list[str] | None = None, bbox: BoundingBox = SAMPLE_BBOX
) -> Dataset:
    url = "http://localhost:9000/datasets/Organization_X/l4_sample"
    return RemoteDataset.open(url).download(
        path=UPath(tmpdir) / "sample_ds", bounding_box=bbox, layers=layers
    )


def attach_agglomerate(seg_layer: SegmentationLayer) -> None:
    seg_data = seg_layer.get_finest_mag().read()
    seg_ids = np.unique(seg_data)

    # Remap segment ids to be dense
    remapped_seg_ids = {seg_id: i + 1 for i, seg_id in enumerate(seg_ids)}
    for old_id, new_id in remapped_seg_ids.items():
        seg_data[seg_data == old_id] = new_id
    seg_layer.get_finest_mag().write(seg_data)

    # Construct agglomerate graph and attachment
    seg_ids = np.unique(seg_data)
    graph = AgglomerateGraph()
    for seg_id in seg_ids:
        seg_position = seg_layer.bounding_box.topleft_xyz + Vec3Int(
            np.argwhere(seg_data == seg_id)[0][1:]
        )
        graph.add_segment(int(seg_id), position=seg_position)
    for seg_id in seg_ids[1:]:
        graph.add_affinity_edge(int(seg_id), int(seg_ids[0]), affinity=0.5)
    AgglomerateAttachment.create_and_add_to(seg_layer, "map_all", graph)


def reopen_dataset(dataset: RemoteDataset) -> RemoteDataset:
    return dataset.reopen(access_mode=RemoteAccessMode.DIRECT_PATH)


@pytest.fixture(scope="module")
def sample_layer_and_mag_name() -> list[tuple[str, str]]:
    layer_names = ["color", "segmentation"]
    mag_names = ["1", "2-2-1", "4-4-1"]
    return list(itertools.product(layer_names, mag_names))


def _prepare_dataset_path(output_path: UPath, suffix: str) -> UPath:
    new_dataset_path = output_path / f"zarr3_dataset_{suffix}"
    rmtree(new_dataset_path)
    return new_dataset_path


@pytest.mark.parametrize(
    "transfer_mode",
    [TransferMode.HTTP, TransferMode.COPY, TransferMode.MOVE_AND_SYMLINK],
)
def test_remote_dataset_add_layer_as_copy(transfer_mode: TransferMode) -> None:
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
    remote_ds = reopen_dataset(remote_ds)
    layer2 = remote_ds.add_layer_as_copy(
        source_layer, new_layer_name="color2", transfer_mode=transfer_mode
    )
    assert len(remote_ds.layers) == 2
    np.testing.assert_array_equal(
        layer2.get_mag(1).read(), source_layer.get_mag(1).read()
    )


@pytest.mark.parametrize(
    "transfer_mode",
    [TransferMode.HTTP, TransferMode.COPY, TransferMode.MOVE_AND_SYMLINK],
)
def test_remote_dataset_add_mag_as_copy(transfer_mode: TransferMode) -> None:
    ds_path = _prepare_dataset_path(TESTOUTPUT_DIR, "remote_copy_mag_src")
    source_ds = Dataset(ds_path, voxel_size=(2, 2, 1))
    source_layer = source_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=DataFormat.Zarr3
    )
    source_mag = source_layer.add_mag(1)
    source_mag.write(
        absolute_offset=(0, 0, 0),
        data=(np.random.rand(16, 16, 16) * 255).astype(np.uint8),
        allow_resize=True,
    )

    remote_ds = source_ds.upload(new_dataset_name="test_remote_dataset_add_mag_as_copy")
    remote_ds = reopen_dataset(remote_ds)
    remote_layer = remote_ds.add_layer(
        "color2", COLOR_CATEGORY, data_format=DataFormat.Zarr3
    )
    copied_mag = remote_layer.add_mag_as_copy(
        source_mag, transfer_mode=transfer_mode, extend_layer_bounding_box=True
    )
    assert Mag(1) in remote_layer.mags
    np.testing.assert_array_equal(copied_mag.read(), source_mag.read())


def test_add_remote_mags_from_mag_view(
    sample_downloaded_dataset: Dataset,
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
        new_layer = sample_downloaded_dataset.add_layer(
            layer_name,
            layer_type,
            data_format=remote_mag.info.data_format,
            dtype=remote_mag.get_dtype(),
        )
        new_layer.add_mag_as_ref(remote_mag)
        added_mag = sample_downloaded_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


def test_add_remote_mags_from_path(
    sample_downloaded_dataset: Dataset,
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
        new_layer = sample_downloaded_dataset.add_layer(
            layer_name,
            layer_type,
            data_format=remote_mag.info.data_format,
            dtype=remote_mag.get_dtype(),
        )
        new_layer.add_mag_as_ref(str(remote_mag.path))
        added_mag = sample_downloaded_dataset.layers[layer_name].mags[remote_mag.mag]
        # checking whether the added_mag.path matches the mag_url with or without a trailing slash.
        assert (
            str(added_mag.path) == str(mag_path)  # or added_mag.path == mag_path.parent
        ), "Added remote mag's path does not match remote path of mag added."


@pytest.mark.parametrize(
    "transfer_mode",
    [TransferMode.HTTP, TransferMode.COPY, TransferMode.MOVE_AND_SYMLINK],
)
def test_remote_attachments_add_attachment_as_copy(
    tmp_upath: UPath, transfer_mode: TransferMode
) -> None:
    local_ds = get_sample_dataset(
        tmp_upath / "source",
        layers=["segmentation"],
        bbox=SAMPLE_BBOX.with_size_xyz(Vec3Int(32, 32, 32)),
    )

    remote_ds = local_ds.upload(new_dataset_name="test_add_layer_as_ref_src")
    remote_ds = reopen_dataset(remote_ds)

    attach_agglomerate(local_ds.get_segmentation_layer("segmentation"))
    remote_ds.get_segmentation_layer("segmentation").attachments.add_attachment_as_copy(
        local_ds.get_segmentation_layer("segmentation").attachments.agglomerates[0],
        transfer_mode=transfer_mode,
    )

    assert (
        len(
            reopen_dataset(remote_ds)
            .get_segmentation_layer("segmentation")
            .attachments.agglomerates
        )
        == 1
    )


def test_ref_layer_from_remote_layer(sample_downloaded_dataset: Dataset) -> None:
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    assert remote_dataset.access_mode == RemoteAccessMode.ZARR_STREAMING
    assert is_remote_path(remote_dataset.get_color_layers()[0].get_mag(1).path), (
        "zarr streaming mag path should be remote."
    )
    sample_remote_layer = list(remote_dataset.layers.values())
    for layer in sample_remote_layer:
        layer_name = f"test_remote_layer_{layer.category}_object"
        sample_downloaded_dataset.add_layer_as_ref(layer, layer_name)
        new_layer = sample_downloaded_dataset.layers[layer_name]
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


def test_explored_remote_access_modes_return_same_data() -> None:
    """Regression test: for an externally explored dataset whose layer is a single
    channel split out of a multi-channel source, ZARR_STREAMING used to return
    different (and briefly, all-zero) data than PROXY_PATH/DIRECT_PATH, because the
    datastore's own served representation for such a layer can legitimately disagree
    with the api's metadata (e.g. channel index, axis order, bounding box) -- see
    idr0062A on webknossos.org, reported in #1492."""
    remote_dataset = RemoteDataset.explore_and_add_remote(
        "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr",
        "idr0062A_access_mode_regression",
        folder=RemoteFolder.get_by_path("Organization_X"),
    )
    try:
        layer = remote_dataset.get_layer("Dapi")
        reference = None
        for access_mode in RemoteAccessMode:
            mag = layer.get_mag(1, access_mode=access_mode)
            data = mag.read()
            # Different access modes may legitimately report different (but equally
            # valid) axis orders for this dataset; normalize to (c, x, y, z) before
            # comparing so only the actual voxel values are asserted to match.
            bbox = mag.normalized_bounding_box
            axis_position = dict(zip(bbox.axes, bbox.index))
            normalized = np.transpose(
                data, [axis_position[a] for a in ("c", "x", "y", "z")]
            )
            if reference is None:
                assert normalized.sum() > 0, (
                    f"{access_mode} unexpectedly returned all-zero data."
                )
                reference = normalized
            else:
                np.testing.assert_array_equal(
                    normalized,
                    reference,
                    err_msg=f"{access_mode} returned different data than "
                    + f"{next(iter(RemoteAccessMode))}.",
                )
    finally:
        remote_dataset.delete()


def test_per_mag_access_mode() -> None:
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    layer = remote_dataset.get_layer("color")

    expected_data_format = {
        RemoteAccessMode.DIRECT_PATH: DataFormat.Zarr3,
        RemoteAccessMode.PROXY_PATH: DataFormat.Zarr3,
        RemoteAccessMode.ZARR_STREAMING: DataFormat.Zarr,
    }
    reference = None
    for access_mode in RemoteAccessMode:
        mag = layer.get_mag(1, access_mode=access_mode)
        assert mag.access_mode == access_mode
        assert mag.data_format == expected_data_format[access_mode]

        data = mag.read(absolute_bounding_box=SAMPLE_BBOX)
        if reference is None:
            reference = data
        else:
            np.testing.assert_array_equal(
                data, reference, f"{access_mode} reads different data."
            )

    # The computed paths follow the datastore routes and are not stored in the properties.
    mag = layer.get_mag(1)
    paths = mag.paths
    assert set(paths.keys()) == set(RemoteAccessMode)
    assert not is_remote_path(paths[RemoteAccessMode.DIRECT_PATH])
    assert str(paths[RemoteAccessMode.ZARR_STREAMING]).endswith("/color/1")
    assert str(paths[RemoteAccessMode.PROXY_PATH]).endswith(
        "/proxy/layers/color/mags/1"
    )
    assert "/zarr/" in str(paths[RemoteAccessMode.ZARR_STREAMING])
    # Testing the immutability of the paths property
    with pytest.raises(TypeError):
        paths[RemoteAccessMode.DIRECT_PATH] = paths[RemoteAccessMode.DIRECT_PATH]  # type: ignore[index]

    # with_access_mode() is equivalent to layer.get_mag(mag, access_mode=...).
    for access_mode in RemoteAccessMode:
        assert (
            mag.with_access_mode(access_mode).path
            == layer.get_mag(1, access_mode=access_mode).path
        )


def test_direct_path_available_in_zarr_streaming_mode() -> None:
    """The underlying paths stay accessible, no matter how the dataset was opened."""
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    assert remote_dataset.access_mode == RemoteAccessMode.ZARR_STREAMING
    for layer in remote_dataset.layers.values():
        for mag in layer.mags.values():
            assert mag.access_mode == RemoteAccessMode.ZARR_STREAMING
            assert RemoteAccessMode.DIRECT_PATH in mag.paths, (
                f"{layer.name}/{mag.name} should expose its direct path."
            )


def test_mixed_access_modes_per_mag() -> None:
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    layer = remote_dataset.get_layer("color")

    mag1 = layer.get_mag(1, access_mode=RemoteAccessMode.DIRECT_PATH)
    mag2 = layer.get_mag("2-2-1", access_mode=RemoteAccessMode.PROXY_PATH)
    assert mag1.access_mode == RemoteAccessMode.DIRECT_PATH
    assert mag2.access_mode == RemoteAccessMode.PROXY_PATH
    assert mag1.read(absolute_bounding_box=SAMPLE_BBOX).sum() > 0
    assert mag2.read(absolute_bounding_box=SAMPLE_BBOX).sum() > 0

    # Requesting a mode explicitly must not change the layer's own mags.
    assert all(
        mag.access_mode == remote_dataset.access_mode for mag in layer.mags.values()
    )
    assert layer.get_mag(1) is not mag1


def test_attachments_are_available_in_every_access_mode(tmp_upath: UPath) -> None:
    local_ds = get_sample_dataset(
        tmp_upath / "source",
        layers=["segmentation"],
        bbox=SAMPLE_BBOX.with_size_xyz(Vec3Int(32, 32, 32)),
    )
    remote_ds = reopen_dataset(
        local_ds.upload(new_dataset_name="test_attachment_access_modes")
    )
    attach_agglomerate(local_ds.get_segmentation_layer("segmentation"))
    remote_ds.get_segmentation_layer("segmentation").attachments.add_attachment_as_copy(
        local_ds.get_segmentation_layer("segmentation").attachments.agglomerates[0]
    )

    # Attachments default to DIRECT_PATH resolution regardless of the dataset's own
    # default access mode, by fetching the api data source on demand.
    streaming_ds = RemoteDataset.open(
        dataset_id=remote_ds.dataset_id, access_mode=RemoteAccessMode.ZARR_STREAMING
    )
    attachments = streaming_ds.get_segmentation_layer("segmentation").attachments
    assert len(attachments.agglomerates) == 1
    assert attachments.access_mode == RemoteAccessMode.DIRECT_PATH

    # PROXY_PATH resolves attachments too, from its own served properties.
    proxy_attachments = attachments.with_access_mode(RemoteAccessMode.PROXY_PATH)
    assert len(proxy_attachments.agglomerates) == 1

    # ZARR_STREAMING never lists attachments at all, so this is simply empty.
    streaming_attachments = attachments.with_access_mode(
        RemoteAccessMode.ZARR_STREAMING
    )
    assert len(streaming_attachments.agglomerates) == 0

    # Attachments are also reachable through the proxy, at a computed path.
    proxy_attachments = attachments.with_access_mode(RemoteAccessMode.PROXY_PATH)
    assert proxy_attachments.access_mode == RemoteAccessMode.PROXY_PATH
    proxy_agglomerate = proxy_attachments.agglomerates[0]
    assert is_remote_path(proxy_agglomerate.path)
    assert str(proxy_agglomerate.path).endswith(
        "/proxy/layers/segmentation/attachments/agglomerate/map_all"
    )
    assert proxy_agglomerate.data_format == attachments.agglomerates[0].data_format


def test_shallow_copy_remote_layers(tmp_upath: UPath) -> None:
    dataset = Dataset(tmp_upath / "origin", voxel_size=(10, 10, 10))
    remote_dataset = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    dataset.add_layer_as_ref(remote_dataset.get_layer("color"), "color")
    dataset_copy = dataset.shallow_copy_dataset(tmp_upath / "copy")
    data = dataset_copy.get_layer("color").get_mag("16-16-4").read()
    assert data.shape == (1, 64, 64, 256)


def test_default_access_mode_reflects_writes_immediately() -> None:
    """Regression test: after a metadata write (here, adding a mag by reference), the
    dataset's own default access mode must be able to read it immediately, not a
    cached copy of that mode's served properties from before the write."""
    ds_path = _prepare_dataset_path(TESTOUTPUT_DIR, "default_mode_write_regression")
    source_ds = Dataset(ds_path, voxel_size=(11.24, 11.24, 28))
    source_layer = source_ds.add_layer(
        "color", COLOR_CATEGORY, data_format=DataFormat.Zarr3
    )
    source_layer.add_mag(1).write(
        absolute_offset=(0, 0, 0),
        data=(np.random.rand(16, 16, 16) * 255).astype(np.uint8),
        allow_resize=True,
    )

    remote_ds = source_ds.upload(new_dataset_name="test_default_mode_write_regression")
    ds = RemoteDataset.open(dataset_id=remote_ds.dataset_id)  # default ZARR_STREAMING
    assert ds.access_mode == RemoteAccessMode.ZARR_STREAMING
    layer = ds.get_layer("color")

    # Populate the cache of ZARR_STREAMING's own served properties before the write.
    assert Mag(1) in layer.mags

    l4_sample = RemoteDataset.open("l4_sample", organization_id="Organization_X")
    layer.add_mag_as_ref(
        l4_sample.get_layer("color").get_mag(1), mag=2, extend_layer_bounding_box=False
    )

    # A freshly resolved mag for the newly added mag 2 must not hit the stale cache.
    assert Mag(2) in layer.mags
    assert layer.get_mag(2).access_mode == RemoteAccessMode.ZARR_STREAMING


def test_changing_properties_on_remote_dataset() -> None:
    """Metadata writes work under the default ZARR_STREAMING access mode too, since
    self._properties is always api-sourced whenever possible, independently of which
    access mode is used to read image data."""
    remote_dataset = RemoteDataset.open(dataset_id="59e9cfbdba632ac2ab8b23b5")
    assert remote_dataset.access_mode == RemoteAccessMode.ZARR_STREAMING
    remote_dataset.description = "This is a test description"
    assert remote_dataset.description == "This is a test description"

    largest_segment_id_before_change = remote_dataset.get_segmentation_layer(
        "segmentation"
    ).largest_segment_id
    remote_dataset.get_segmentation_layer("segmentation").largest_segment_id = 10
    assert (
        remote_dataset.get_segmentation_layer("segmentation").largest_segment_id == 10
    )
    remote_dataset.get_segmentation_layer(
        "segmentation"
    ).largest_segment_id = largest_segment_id_before_change

    default_view_configuration_before_change = remote_dataset.get_layer(
        "color"
    ).default_view_configuration
    new_view_configuration = LayerViewConfiguration(alpha=30.0)
    remote_dataset.get_layer(
        "color"
    ).default_view_configuration = new_view_configuration
    assert (
        remote_dataset.get_layer("color").default_view_configuration
        == new_view_configuration
    )
    remote_dataset.get_layer(
        "color"
    ).default_view_configuration = default_view_configuration_before_change


def test_changing_properties_fails_without_direct_sourced_properties() -> None:
    """Metadata writes are rejected when there is no api-sourced properties to write
    back to, e.g. an annotation's volume layers, which are not part of the underlying
    dataset's api data source. This is independent of access mode (see
    test_changing_properties_on_remote_dataset)."""
    path = TESTDATA_DIR / "annotations" / "l4_sample__explorational__suser__94b271.zip"
    annotation_from_file = Annotation.load(path)
    annotation_from_file.organization_id = "Organization_X"
    annotation = Annotation.download(annotation_from_file.upload())

    ds = annotation.get_remote_annotation_dataset()
    assert ds.access_mode == RemoteAccessMode.ZARR_STREAMING
    layer = ds.layers["Volume"]
    default_view_configuration_before_attempt = layer.default_view_configuration
    with pytest.raises(RuntimeError):
        layer.default_view_configuration = LayerViewConfiguration(alpha=0.3)
    assert layer.default_view_configuration == default_view_configuration_before_attempt


def test_remote_layer_view_configuration() -> None:
    remote_dataset = RemoteDataset.open(dataset_id="59e9cfbdba632ac2ab8b23b5")
    layer = remote_dataset.get_layer("color")
    config = layer.view_configuration

    assert isinstance(config, LayerViewConfiguration)
    assert config.alpha == 100.0
    assert config.is_inverted == False

    # config values are readyonly
    with pytest.raises(AttributeError):
        layer.view_configuration = LayerViewConfiguration(alpha=50.0)


def test_remote_layer_coordinate_transformations(tmp_upath: UPath) -> None:
    local_dataset = get_sample_dataset(tmp_upath / "source", layers=["color"])
    remote_dataset = reopen_dataset(
        local_dataset.upload(new_dataset_name="test_coordinate_transformations")
    )

    layer = remote_dataset.get_layer("color")
    assert layer.coordinate_transformations == ()

    affine = AffineCoordinateTransformation.from_translation((10, 20, 30))
    thin_plate_spline = ThinPlateSplineCoordinateTransformation(
        source=[[0, 0, 0], [1, 2, 3], [4, 5, 6]],
        target=[[1, 1, 1], [2, 4, 6], [8, 10, 12]],
    )
    layer.coordinate_transformations = [affine, thin_plate_spline]
    assert layer.coordinate_transformations == (affine, thin_plate_spline)

    # Test if the transformations are persisted on the server
    assert reopen_dataset(remote_dataset).get_layer(
        "color"
    ).coordinate_transformations == (affine, thin_plate_spline)

    layer.coordinate_transformations = []
    assert layer.coordinate_transformations == ()
    assert (
        reopen_dataset(remote_dataset).get_layer("color").coordinate_transformations
        == ()
    )


def test_changing_properties_on_remote_layer_with_zarr_streaming() -> None:
    """ZARR_STREAMING loads self._properties from the api data source too (just not the
    per-mag reads, see test_per_mag_access_mode), so metadata writes work here as well."""
    remote_dataset = RemoteDataset.open(dataset_id="59e9cfbdba632ac2ab8b23b5")
    assert remote_dataset.access_mode == RemoteAccessMode.ZARR_STREAMING
    layer = remote_dataset.get_layer("color")
    coordinate_transformations_before_change = layer.coordinate_transformations
    new_coordinate_transformations = [
        AffineCoordinateTransformation.from_translation((10, 20, 30))
    ]
    layer.coordinate_transformations = new_coordinate_transformations
    assert layer.coordinate_transformations == tuple(new_coordinate_transformations)
    layer.coordinate_transformations = coordinate_transformations_before_change


def test_changing_properties_on_read_only_remote_dataset() -> None:
    remote_dataset = RemoteDataset.open(
        dataset_id="59e9cfbdba632ac2ab8b23b5", read_only=True
    )
    description_before_change_attempt = remote_dataset.description
    with pytest.raises(RuntimeError):
        remote_dataset.description = "This is a test description"
    assert remote_dataset.description == description_before_change_attempt


@pytest.mark.parametrize(
    "access_mode", [RemoteAccessMode.ZARR_STREAMING, RemoteAccessMode.DIRECT_PATH]
)
def test_read_only_forces_layers_read_only(access_mode: RemoteAccessMode) -> None:
    """Metadata writes normally work under any access mode, but read_only=True must
    still win, regardless of which access mode is used to read image data."""
    read_only_dataset = RemoteDataset.open(
        dataset_id="59e9cfbdba632ac2ab8b23b5",
        access_mode=access_mode,
        read_only=True,
    )
    layer = read_only_dataset.get_layer("color")
    assert layer.read_only
    default_view_configuration_before_change_attempt = layer.default_view_configuration
    with pytest.raises(RuntimeError):
        layer.default_view_configuration = LayerViewConfiguration(alpha=0.3)
    assert (
        layer.default_view_configuration
        == default_view_configuration_before_change_attempt
    )

    writable_dataset = RemoteDataset.open(
        dataset_id="59e9cfbdba632ac2ab8b23b5", access_mode=access_mode
    )
    assert not writable_dataset.get_layer("color").read_only


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
    ],
)
def test_url_download(url: str, tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    ds = RemoteDataset.open(url).download(
        path=tmp_upath / "ds", mags=[Mag(1)], bounding_box=SAMPLE_BBOX
    )
    assert set(ds.layers.keys()) == {"color", "segmentation"}
    data = ds.get_color_layers()[0].get_finest_mag().read()
    assert data.sum() == 120697
    np.testing.assert_array_equal(
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
@pytest.mark.parametrize("access_mode", list(RemoteAccessMode))
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
    np.testing.assert_array_equal(
        data,
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )
    assert set(pickle.loads(pickle.dumps(ds)).layers.keys()) == {
        "color",
        "segmentation",
    }, "Dataset instances should be picklable."


@pytest.mark.parametrize(
    "transfer_mode", [TransferMode.COPY, TransferMode.MOVE_AND_SYMLINK]
)
@pytest.mark.parametrize("api_version", [13, 14])
def test_upload_dataset(
    tmp_upath: UPath, transfer_mode: TransferMode, api_version: int
) -> None:
    from webknossos.client.context import _get_context

    with webknossos_context(
        url=_get_context().url, token=_get_context().token, api_version=api_version
    ):
        sample_dataset = get_sample_dataset(tmp_upath)
        remote_ds = sample_dataset.upload(
            new_dataset_name="test_remote_symlink",
            transfer_mode=transfer_mode,
        )
        np.testing.assert_array_equal(
            remote_ds.get_color_layers()[0].get_finest_mag().read(),
            sample_dataset.get_color_layers()[0].get_finest_mag().read(),
        )


def test_remote_dataset(tmp_upath: UPath) -> None:
    sample_dataset = get_sample_dataset(tmp_upath)
    remote_ds = sample_dataset.upload(new_dataset_name="test_remote_metadata")
    np.testing.assert_array_equal(
        remote_ds.get_color_layers()[0].get_finest_mag().read(),
        sample_dataset.get_color_layers()[0].get_finest_mag().read(),
    )

    assert not remote_ds.read_only
    # Metadata is writable, since self._properties is api-sourced.
    assert not remote_ds.get_color_layers()[0].read_only
    # The mag's array data itself is always read-only for remote layers, though.
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
    ds_roundtrip = RemoteDataset.open(uploaded_dataset.url).download(
        path=tmp_upath / "ds", layers=["color", "segmentation"]
    )
    assert set(ds_original.get_segmentation_layers()[0].mags.keys()) == set(
        ds_roundtrip.get_segmentation_layers()[0].mags.keys()
    )

    data_original = ds_original.get_segmentation_layers()[0].get_finest_mag().read()
    data_roundtrip = ds_roundtrip.get_segmentation_layers()[0].get_finest_mag().read()
    np.testing.assert_array_equal(data_original, data_roundtrip)


def test_upload_twice(tmp_upath: UPath) -> None:
    ds_original = get_sample_dataset(tmp_upath)
    remote1 = ds_original.upload(new_dataset_name="test_upload_twice")
    remote2 = ds_original.upload(new_dataset_name="test_upload_twice")
    assert remote1.url != remote2.url
    assert remote1.name == remote2.name


def test_delete_remote_dataset(tmp_upath: UPath) -> None:
    ds_original = get_sample_dataset(tmp_upath)
    remote_ds = ds_original.upload(new_dataset_name="test_delete_remote_dataset")
    dataset_id = remote_ds.dataset_id

    remote_ds.delete()

    assert remote_ds.read_only
    with pytest.raises(RuntimeError):
        remote_ds.description = "should fail"
    assert dataset_id not in RemoteDataset.list()


def test_remote_dataset_add_layer_as_ref_rejects_local_layer(
    tmp_upath: UPath,
) -> None:
    """add_layer_as_ref raises ValueError when given a local (non-remote) layer."""
    local_ds = Dataset(tmp_upath / "local", voxel_size=(1, 1, 1))
    local_layer = local_ds.add_layer("color", COLOR_CATEGORY)

    mock_ds = MagicMock()
    mock_ds.layers = {}

    with pytest.raises(ValueError, match="local layer"):
        RemoteDataset.add_layer_as_ref(mock_ds, local_layer)


def test_remote_dataset_add_layer_as_ref_rejects_existing_layer_name() -> None:
    """add_layer_as_ref raises IndexError when the target already has a layer with that name."""
    mock_foreign_layer = MagicMock(spec=RemoteLayer)
    mock_foreign_layer.name = "color"
    mock_foreign_layer.dataset = MagicMock()  # a different dataset

    mock_ds = MagicMock()
    mock_ds.layers = {"color": MagicMock()}  # "color" already present

    with pytest.raises(IndexError, match="already has a layer"):
        RemoteDataset.add_layer_as_ref(mock_ds, mock_foreign_layer)


def test_remote_dataset_add_layer_as_ref_rejects_same_dataset() -> None:
    """add_layer_as_ref raises ValueError when the foreign layer belongs to the same dataset."""
    mock_ds = MagicMock()
    mock_ds.layers = {}

    mock_foreign_layer = MagicMock(spec=RemoteLayer)
    mock_foreign_layer.name = "color"
    mock_foreign_layer.dataset = mock_ds  # same object → equality check fires

    with pytest.raises(ValueError, match="same origin dataset"):
        RemoteDataset.add_layer_as_ref(mock_ds, mock_foreign_layer)


def test_remote_dataset_add_layer_as_ref_rejects_different_instance() -> None:
    """add_layer_as_ref raises ValueError when the foreign layer is from a different WK instance."""
    mock_ds = MagicMock()
    mock_ds.layers = {}
    mock_ds._context._url = "http://instance-a:9000"

    mock_foreign_layer = MagicMock(spec=RemoteLayer)
    mock_foreign_layer.name = "color"
    mock_foreign_layer.dataset._context._url = "http://instance-b:9000"
    mock_foreign_layer.dataset.__eq__ = lambda _self, _other: False  # different dataset

    with pytest.raises(ValueError, match="different WEBKNOSSOS instance"):
        RemoteDataset.add_layer_as_ref(mock_ds, mock_foreign_layer)


def test_remote_layer_add_mag_as_ref_rejects_local_mag(tmp_upath: UPath) -> None:
    """add_mag_as_ref raises ValueError when given a local (non-remote) MagView."""
    local_mag = (
        Dataset(tmp_upath / "local", voxel_size=(1, 1, 1))
        .add_layer("color", COLOR_CATEGORY)
        .add_mag(Mag(1))
    )

    mock_remote_layer = MagicMock(spec=RemoteLayer)

    with pytest.raises(ValueError, match="local mag"):
        RemoteLayer.add_mag_as_ref(mock_remote_layer, local_mag)


def test_remote_layer_add_mag_as_ref_rejects_different_instance() -> None:
    """add_mag_as_ref raises ValueError when the foreign mag is from a different WK instance."""
    from unittest.mock import patch

    from webknossos.dataset.layer.view.mag_view import MagView

    mock_remote_layer = MagicMock(spec=RemoteLayer)
    mock_remote_layer.dataset._context._url = "http://instance-a:9000"

    mock_foreign_layer = MagicMock(spec=RemoteLayer)
    mock_foreign_layer.dataset._context._url = "http://instance-b:9000"

    mock_foreign_mag = MagicMock(spec=MagView)
    mock_foreign_mag.layer = mock_foreign_layer

    with patch.object(
        MagView, "_ensure_mag_view", staticmethod(lambda _x: mock_foreign_mag)
    ):
        with pytest.raises(ValueError, match="different WEBKNOSSOS instance"):
            RemoteLayer.add_mag_as_ref(mock_remote_layer, mock_foreign_mag)


def test_remote_attachments_add_attachment_as_ref_rejects_different_instance() -> None:
    """add_attachment_as_ref raises ValueError when foreign_layer is from a different WK instance."""
    from webknossos.dataset.layer.segmentation_layer import RemoteSegmentationLayer

    mock_layer = MagicMock()
    mock_layer.dataset._context._url = "http://instance-a:9000"

    mock_foreign_layer = MagicMock(spec=RemoteSegmentationLayer)
    mock_foreign_layer.dataset._context._url = "http://instance-b:9000"

    mock_attachments = MagicMock(spec=RemoteAttachments)
    mock_attachments._layer = mock_layer

    attachment = MagicMock(spec=AgglomerateAttachment)
    attachment.name = "identity"

    with pytest.raises(ValueError, match="different WEBKNOSSOS instance"):
        RemoteAttachments.add_attachment_as_ref(
            mock_attachments,
            attachment,
            foreign_layer=mock_foreign_layer,
            new_name=None,
        )


def test_remote_attachments_rename_attachment_rejects_invalid_name() -> None:
    """rename_attachment raises ValueError for names with invalid characters."""
    mock_layer = MagicMock()
    mock_layer._ensure_writable.return_value = None

    mock_attachments = MagicMock(spec=RemoteAttachments)
    mock_attachments._layer = mock_layer

    attachment = MagicMock(spec=AgglomerateAttachment)
    attachment.name = "agglomerate_view_75"

    with pytest.raises(ValueError, match="not allowed"):
        RemoteAttachments.rename_attachment(
            mock_attachments,
            attachment,
            new_name="invalid/name",
        )


# --- Happy-path tests against server ---


def test_remote_dataset_add_layer_and_mag_as_ref(tmp_upath: UPath) -> None:
    """add_layer_as_ref creates a reference to a layer from another remote dataset."""
    source_local = get_sample_dataset(tmp_upath / "source", layers=["color"])
    source_remote = source_local.upload(new_dataset_name="test_add_layer_as_ref_src")

    source_remote = reopen_dataset(source_remote)

    target_local = get_sample_dataset(tmp_upath / "target", layers=["color"])
    target_remote = target_local.upload(new_dataset_name="test_add_layer_as_ref_tgt")

    target_remote = reopen_dataset(target_remote)

    # add_layer_as_ref
    source_color = source_remote.get_layer("color")
    new_layer = target_remote.add_layer_as_ref(source_color, new_layer_name="color_ref")

    assert "color_ref" in target_remote.layers
    assert new_layer.name == "color_ref"
    assert str(new_layer.get_mag(Mag(1)).path) == str(source_color.get_mag(Mag(1)).path)
    assert set(new_layer.mags.keys()) == {
        Mag(1),
        Mag((2, 2, 1)),
        Mag((4, 4, 1)),
        Mag((8, 8, 2)),
        Mag((16, 16, 4)),
    }
    assert "color_ref" in reopen_dataset(target_remote).layers

    # delete_mag
    new_layer.delete_mag(Mag((2, 2, 1)))
    new_layer.delete_mag(Mag((4, 4, 1)))
    new_layer.delete_mag(Mag((8, 8, 2)))
    new_layer.delete_mag(Mag((16, 16, 4)))
    assert set(new_layer.mags.keys()) == {Mag(1)}
    assert set(reopen_dataset(target_remote).get_layer("color_ref").mags) == set(
        new_layer.mags.keys()
    )

    # add_mag_as_ref
    new_layer.add_mag_as_ref(source_color.get_mag(Mag((2, 2, 1))))
    assert set(new_layer.mags.keys()) == {Mag(1), Mag((2, 2, 1))}
    assert set(reopen_dataset(target_remote).get_layer("color_ref").mags) == set(
        new_layer.mags.keys()
    )

    # rename layer
    new_layer.name = "color_ref_renamed"
    assert new_layer.name == "color_ref_renamed"
    assert "color_ref_renamed" in target_remote.layers.keys()
    assert "color_ref_renamed" in reopen_dataset(target_remote).layers

    # delete layer
    target_remote.delete_layer("color_ref_renamed")
    assert "color_ref_renamed" not in target_remote.layers.keys()
    assert "color_ref_renamed" not in reopen_dataset(target_remote).layers


@pytest.mark.parametrize(
    "transfer_mode",
    [TransferMode.HTTP, TransferMode.COPY, TransferMode.MOVE_AND_SYMLINK],
)
def test_remote_attachments_add_attachment_as_ref(
    tmp_upath: UPath, transfer_mode: TransferMode
) -> None:
    """add_attachment_as_ref references an attachment from another remote segmentation layer."""
    source_local = get_sample_dataset(
        tmp_upath / "source",
        layers=["segmentation"],
        bbox=SAMPLE_BBOX.with_size_xyz(Vec3Int(32, 32, 32)),
    )
    attach_agglomerate(source_local.get_segmentation_layer("segmentation"))
    source_remote = source_local.upload(new_dataset_name="test_add_layer_as_ref_src")
    source_remote = reopen_dataset(source_remote)

    target_local = get_sample_dataset(tmp_upath / "target", layers=["color"])
    target_remote = target_local.upload(
        new_dataset_name="test_add_layer_as_ref_tgt", transfer_mode=transfer_mode
    )
    target_remote = reopen_dataset(target_remote)

    # Add layer including attachments
    new_layer = target_remote.add_layer_as_ref(
        source_remote.get_layer("segmentation")
    ).as_segmentation_layer()
    assert "segmentation" in target_remote.layers.keys()
    assert len(new_layer.attachments.agglomerates) == 1
    assert (
        len(
            reopen_dataset(target_remote)
            .get_segmentation_layer("segmentation")
            .attachments.agglomerates
        )
        == 1
    )

    # Rename attachment
    new_layer.attachments.rename_attachment(
        new_layer.attachments.agglomerates[0],
        new_name="renamed_agglomerate",
    )
    assert new_layer.attachments.agglomerates[0].name == "renamed_agglomerate"
    assert (
        reopen_dataset(target_remote)
        .get_segmentation_layer("segmentation")
        .attachments.agglomerates[0]
        .name
        == "renamed_agglomerate"
    )

    # Delete attachment
    new_layer.attachments.delete_attachment(new_layer.attachments.agglomerates[0])
    assert len(new_layer.attachments.agglomerates) == 0
    assert (
        len(
            reopen_dataset(target_remote)
            .get_segmentation_layer("segmentation")
            .attachments.agglomerates
        )
        == 0
    )

    # Add attachment
    new_layer.attachments.add_attachment_as_ref(
        source_remote.get_layer("segmentation").attachments.agglomerates[0],
        foreign_layer=source_remote.get_layer("segmentation"),
        new_name="new_agglomerate",
    )
    assert len(new_layer.attachments.agglomerates) == 1
    assert new_layer.attachments.agglomerates[0].name == "new_agglomerate"
    assert (
        reopen_dataset(target_remote)
        .get_segmentation_layer("segmentation")
        .attachments.agglomerates[0]
        .name
        == "new_agglomerate"
    )

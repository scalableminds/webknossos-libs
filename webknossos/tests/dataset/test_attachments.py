import json
from pathlib import Path

import pytest
from upath import UPath

from webknossos.dataset import (
    SEGMENTATION_CATEGORY,
    AttachmentDataFormat,
    Dataset,
    MeshAttachment,
    SegmentationLayer,
    SegmentIndexAttachment,
)
from webknossos.geometry import BoundingBox


def make_dataset(dataset_path: UPath) -> tuple[Dataset, SegmentationLayer]:
    dataset = Dataset(dataset_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()
    return dataset, seg_layer


def test_attachments(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    # meshes
    seg_layer.attachments.add_mesh(
        dataset.path / "seg" / "meshfile",
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )
    assert seg_layer._properties.attachments.meshes is not None
    assert seg_layer._properties.attachments.meshes[0].path == "./seg/meshfile"
    assert (
        seg_layer._properties.attachments.meshes[0].data_format
        == AttachmentDataFormat.Zarr3
    )
    assert len(seg_layer._properties.attachments.meshes) == 1

    # agglomerates
    seg_layer.attachments.add_agglomerate(
        UPath(
            "s3://bucket/agglomerate.zarr",
            client_kwargs={"endpoint_url": "https://s3.eu-central-1.amazonaws.com"},
        ),
        name="identity",
        data_format=AttachmentDataFormat.Zarr3,
    )
    assert seg_layer._properties.attachments.agglomerates is not None
    assert (
        seg_layer._properties.attachments.agglomerates[0].path
        == "s3://s3.eu-central-1.amazonaws.com/bucket/agglomerate.zarr"
    )
    assert (
        seg_layer._properties.attachments.agglomerates[0].data_format
        == AttachmentDataFormat.Zarr3
    )
    assert len(seg_layer._properties.attachments.agglomerates) == 1

    # connectomes
    seg_layer.attachments.add_connectome(
        UPath("http://example.com/connectome.zarr"),
        name="connectome",
        data_format=AttachmentDataFormat.Zarr3,
    )
    assert seg_layer._properties.attachments.connectomes is not None
    assert (
        seg_layer._properties.attachments.connectomes[0].path
        == "http://example.com/connectome.zarr"
    )
    assert (
        seg_layer._properties.attachments.connectomes[0].data_format
        == AttachmentDataFormat.Zarr3
    )
    assert len(seg_layer._properties.attachments.connectomes) == 1

    # segment index
    seg_layer.attachments.set_segment_index(
        Path.home() / "segment_index.hdf5",
        name="main",
        data_format=AttachmentDataFormat.HDF5,
    )
    assert seg_layer._properties.attachments.segment_index is not None
    assert (
        seg_layer._properties.attachments.segment_index.path
        == (Path.home() / "segment_index.hdf5").as_posix()
    )
    assert (
        seg_layer._properties.attachments.segment_index.data_format
        == AttachmentDataFormat.HDF5
    )

    # cumsum
    seg_layer.attachments.set_cumsum(
        dataset.path / "seg" / "cumsum.json",
        name="main",
        data_format=AttachmentDataFormat.JSON,
    )
    assert seg_layer._properties.attachments.cumsum is not None
    assert seg_layer._properties.attachments.cumsum.path == "./seg/cumsum.json"
    assert (
        seg_layer._properties.attachments.cumsum.data_format
        == AttachmentDataFormat.JSON
    )

    attachments_json = json.loads(
        (dataset.path / "datasource-properties.json").read_text()
    )["dataLayers"][0]["attachments"]
    print(attachments_json)
    assert len(attachments_json["meshes"]) == 1
    assert attachments_json["meshes"][0] == {
        "name": "meshfile",
        "dataFormat": "zarr3",
        "path": "./seg/meshfile",
    }
    assert len(attachments_json["agglomerates"]) == 1
    assert len(attachments_json["connectomes"]) == 1
    assert attachments_json["segmentIndex"] is not None
    assert attachments_json["cumsum"] is not None

    # delete
    seg_layer.attachments.delete_attachment(seg_layer.attachments.meshes[0])
    assert seg_layer._properties.attachments.meshes is None
    seg_layer.attachments.delete_attachment(seg_layer.attachments.agglomerates[0])
    assert seg_layer._properties.attachments.agglomerates is None
    seg_layer.attachments.delete_attachment(seg_layer.attachments.connectomes[0])
    assert seg_layer._properties.attachments.connectomes is None
    seg_layer.attachments.delete_attachment(seg_layer.attachments.segment_index)
    assert seg_layer._properties.attachments.segment_index is None
    seg_layer.attachments.delete_attachment(seg_layer.attachments.cumsum)
    assert seg_layer._properties.attachments.cumsum is None

    properties_json = json.loads(
        (dataset.path / "datasource-properties.json").read_text()
    )
    assert "attachments" not in properties_json["dataLayers"][0]


def test_copy_layer(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    agglomerate_path = (tmp_upath / "agglomerate_view_15").resolve()
    agglomerate_path.mkdir(parents=True, exist_ok=True)
    (agglomerate_path / "zarr.json").write_text("test")

    seg_layer.attachments.add_mesh(
        mesh_path,
        name="meshfile",
        data_format=AttachmentDataFormat.HDF5,
    )
    seg_layer.attachments.add_agglomerate(
        Path("../agglomerate_view_15"),
        name="agglomerate_view_15",
        data_format=AttachmentDataFormat.Zarr3,
    )

    copy_dataset = Dataset(tmp_upath / "test_copy", voxel_size=(10, 10, 10))
    copy_layer = copy_dataset.add_layer_as_copy(seg_layer).as_segmentation_layer()

    assert (
        copy_layer.attachments.meshes[0].path
        == copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    )
    assert (copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5").exists()
    assert (
        copy_layer.attachments.meshes[0]._properties.path
        == "./seg/meshes/meshfile.hdf5"
    )

    assert (
        copy_layer.attachments.agglomerates[0].path
        == copy_dataset.path / "seg" / "agglomerates" / "agglomerate_view_15"
    )
    assert (
        copy_dataset.path / "seg" / "agglomerates" / "agglomerate_view_15" / "zarr.json"
    ).exists()
    assert (
        copy_layer.attachments.agglomerates[0]._properties.path
        == "./seg/agglomerates/agglomerate_view_15"
    )


def test_symlink_layer(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    agglomerate_path = (tmp_upath / "agglomerate_view_15").resolve()
    agglomerate_path.mkdir(parents=True, exist_ok=True)
    (agglomerate_path / "zarr.json").write_text("test")

    seg_layer.attachments.add_mesh(
        mesh_path,
        name="meshfile",
        data_format=AttachmentDataFormat.HDF5,
    )

    seg_layer.attachments.add_agglomerate(
        Path("../agglomerate_view_15"),
        name="agglomerate_view_15",
        data_format=AttachmentDataFormat.Zarr3,
    )

    copy_dataset = Dataset(tmp_upath / "test_copy", voxel_size=(10, 10, 10))
    with pytest.warns(DeprecationWarning):
        copy_layer = copy_dataset.add_symlink_layer(
            seg_layer, make_relative=True
        ).as_segmentation_layer()

    # has been copied
    assert (
        copy_layer.attachments.meshes[0].path
        == dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    )
    assert copy_layer._properties.attachments.meshes is not None
    assert (
        copy_layer._properties.attachments.meshes[0].path
        == "../test_attachments/seg/meshes/meshfile.hdf5"
    )

    # has not been copied
    assert copy_layer.attachments.agglomerates[0].path == agglomerate_path
    assert (
        copy_layer.attachments.agglomerates[0]._properties.path
        == "../agglomerate_view_15"
    )


def test_remote_layer(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    mesh_path = UPath(
        "s3://bucket/meshfile.zarr",
        client_kwargs={"endpoint_url": "https://s3.eu-central-1.amazonaws.com"},
    )

    seg_layer.attachments.add_mesh(
        mesh_path,
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    copy_dataset = Dataset(tmp_upath / "test_copy", voxel_size=(10, 10, 10))
    copy_layer = copy_dataset.add_layer_as_ref(seg_layer).as_segmentation_layer()

    assert copy_layer.attachments.meshes[0].path == mesh_path
    assert copy_layer._properties.attachments.meshes is not None
    assert (
        copy_layer._properties.attachments.meshes[0].path
        == "s3://s3.eu-central-1.amazonaws.com/bucket/meshfile.zarr"
    )


def test_upload_fail(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)
    seg_layer.attachments.add_mesh(
        dataset.path / "seg" / "meshfile",
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    with pytest.raises(NotImplementedError):
        dataset.upload()


def test_unique_attachment_names(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    seg_layer.attachments.add_mesh(
        UPath("http://example.com/meshfile.zarr"),
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )
    with pytest.raises(ValueError):
        seg_layer.attachments.add_mesh(
            UPath("http://example.com/meshfile.zarr"),
            name="meshfile",
            data_format=AttachmentDataFormat.Zarr3,
        )


def test_acceptable_attachment_names(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    with pytest.raises(ValueError):
        seg_layer.attachments.add_mesh(
            UPath("http://example.com/meshfile.zarr"),
            name="meshfile/test",
            data_format=AttachmentDataFormat.Zarr3,
        )


def test_remote_dataset() -> None:
    dataset, seg_layer = make_dataset(UPath("memory://test_attachments"))

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    seg_layer.attachments.add_mesh(
        mesh_path,
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    assert seg_layer.attachments.meshes[0]._properties.path == "./seg/meshes/meshfile"


def test_add_attachments(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    mesh = MeshAttachment.from_path_and_name(
        dataset.path / "seg" / "meshes" / "meshfile",
        "meshfile_4-4-1",
        data_format=AttachmentDataFormat.Zarr3,
    )
    seg_layer.attachments.add_attachment_as_ref(mesh)
    assert seg_layer._properties.attachments.meshes is not None
    assert seg_layer._properties.attachments.meshes[0].path == "./seg/meshes/meshfile"
    assert seg_layer.attachments.meshes[0].name == "meshfile_4-4-1"


def test_add_copy_attachments(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    # meshes
    mesh_path = tmp_upath / "meshfile"
    mesh_path.write_text("test")

    mesh = MeshAttachment.from_path_and_name(
        mesh_path,
        "meshfile_4-4-1",
        data_format=AttachmentDataFormat.Zarr3,
    )
    seg_layer.attachments.add_attachment_as_copy(mesh)
    assert seg_layer._properties.attachments.meshes is not None
    assert seg_layer.attachments.meshes[0].name == "meshfile_4-4-1"
    # path has changed based on the name
    assert (
        seg_layer._properties.attachments.meshes[0].path
        == "./seg/meshes/meshfile_4-4-1"
    )
    assert (dataset.path / "seg" / "meshes" / "meshfile_4-4-1").exists()
    assert mesh_path.exists()

    # segment index (has camel-casing)
    segment_index_path = tmp_upath / "segment_index"
    segment_index_path.write_text("test")

    segment_index = SegmentIndexAttachment.from_path_and_name(
        segment_index_path,
        "main",
        data_format=AttachmentDataFormat.Zarr3,
    )
    seg_layer.attachments.add_attachment_as_copy(segment_index)
    assert seg_layer._properties.attachments.segment_index is not None
    assert (
        seg_layer._properties.attachments.segment_index.path
        == "./seg/segmentIndex/main"
    )


def test_add_symlink_attachments(tmp_upath: UPath) -> None:
    dataset, seg_layer = make_dataset(tmp_upath)

    # meshes
    mesh_path = tmp_upath.resolve() / "meshfile"
    mesh_path.write_text("test")

    mesh = MeshAttachment.from_path_and_name(
        mesh_path,
        "meshfile_4-4-1",
        data_format=AttachmentDataFormat.Zarr3,
    )
    with pytest.warns(DeprecationWarning):
        seg_layer.attachments.add_symlink_attachments(mesh)
    assert seg_layer._properties.attachments.meshes is not None
    assert seg_layer._properties.attachments.meshes[0].path == mesh_path.as_posix()
    assert seg_layer.attachments.meshes[0].name == "meshfile_4-4-1"
    assert (dataset.path / "seg" / "meshes" / "meshfile_4-4-1").exists()
    assert (dataset.path / "seg" / "meshes" / "meshfile_4-4-1").is_symlink()
    assert (dataset.path / "seg" / "meshes" / "meshfile_4-4-1").resolve() == mesh_path
    assert mesh_path.exists()

    # segment index (has camel-casing)
    segment_index_path = tmp_upath / "segment_index"
    segment_index_path.write_text("test")

    segment_index = SegmentIndexAttachment.from_path_and_name(
        segment_index_path,
        "main",
        data_format=AttachmentDataFormat.Zarr3,
    )
    with pytest.warns(DeprecationWarning):
        seg_layer.attachments.add_symlink_attachments(segment_index)
    assert seg_layer._properties.attachments.segment_index is not None
    assert (
        seg_layer._properties.attachments.segment_index.path
        == segment_index_path.as_posix()
    )
    assert (dataset.path / "seg" / "segmentIndex" / "main").exists()
    assert (dataset.path / "seg" / "segmentIndex" / "main").is_symlink()
    assert (
        dataset.path / "seg" / "segmentIndex" / "main"
    ).resolve() == segment_index_path


def test_detect_legacy_attachments(tmp_upath: UPath) -> None:
    _, seg_layer = make_dataset(tmp_upath)

    # legacy meshes
    mesh_path = seg_layer.path / "meshes" / "meshfile_4-4-1.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    # legacy agglomerates
    agglomerate_path = seg_layer.path / "agglomerates" / "agglomerate_view_15.hdf5"
    agglomerate_path.parent.mkdir(parents=True, exist_ok=True)
    agglomerate_path.write_text("test")

    # legacy cumsum
    cumsum_path = seg_layer.path / "agglomerates" / "cumsum.json"
    cumsum_path.parent.mkdir(parents=True, exist_ok=True)
    cumsum_path.write_text("test")

    # legacy agglomerates
    connectome_path = seg_layer.path / "connectomes" / "paper_l4_full_connectome.hdf5"
    connectome_path.parent.mkdir(parents=True, exist_ok=True)
    connectome_path.write_text("test")

    # legacy segment index
    segment_index_path = seg_layer.path / "segmentIndex" / "segmentIndex.hdf5"
    segment_index_path.parent.mkdir(parents=True, exist_ok=True)
    segment_index_path.write_text("test")

    seg_layer.attachments.detect_legacy_attachments()

    assert seg_layer.attachments.meshes[0].path == mesh_path
    assert seg_layer.attachments.agglomerates[0].path == agglomerate_path
    assert seg_layer.attachments.connectomes[0].path == connectome_path

    assert (
        seg_layer.attachments.cumsum
        and seg_layer.attachments.cumsum.path == cumsum_path
    )
    assert (
        seg_layer.attachments.segment_index
        and seg_layer.attachments.segment_index.path == segment_index_path
    )

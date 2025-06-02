import json
from pathlib import Path

import pytest
from upath import UPath

from webknossos.dataset import (
    SEGMENTATION_CATEGORY,
    AttachmentDataFormat,
    Dataset,
)
from webknossos.geometry import BoundingBox


def test_attachments(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

    # meshes
    seg_layer.attachments.add_mesh(
        dataset.path / "seg" / "meshfile",
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )
    assert seg_layer._properties.attachments.meshes is not None
    assert seg_layer._properties.attachments.meshes[0].path == "seg/meshfile"
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
        "/usr/segment_index.hdf5",
        name="main",
        data_format=AttachmentDataFormat.HDF5,
    )
    assert seg_layer._properties.attachments.segment_index is not None
    assert (
        seg_layer._properties.attachments.segment_index.path
        == "/usr/segment_index.hdf5"
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
    assert seg_layer._properties.attachments.cumsum.path == "seg/cumsum.json"
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
        "path": "seg/meshfile",
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


def test_copy_layer(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    agglomerate_path = (tmp_path / "agglomerate_view_15").resolve()
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

    copy_dataset = Dataset(tmp_path / "test_copy", voxel_size=(10, 10, 10))
    copy_layer = copy_dataset.add_copy_layer(seg_layer).as_segmentation_layer()

    assert (
        copy_layer.attachments.meshes[0].path
        == copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    )
    assert (copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5").exists()
    assert (
        copy_layer.attachments.meshes[0]._properties.path == "seg/meshes/meshfile.hdf5"
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
        == "seg/agglomerates/agglomerate_view_15"
    )


def test_fs_copy_layer(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    agglomerate_path = (tmp_path / "agglomerate_view_15").resolve()
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

    copy_dataset = Dataset(tmp_path / "test_copy", voxel_size=(10, 10, 10))
    copy_layer = copy_dataset.add_fs_copy_layer(seg_layer).as_segmentation_layer()

    # has been copied
    assert (
        copy_layer.attachments.meshes[0].path
        == copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    )
    assert (copy_dataset.path / "seg" / "meshes" / "meshfile.hdf5").exists()

    # has not been copied
    assert copy_layer.attachments.agglomerates[0].path == agglomerate_path
    assert copy_layer.attachments.agglomerates[0]._properties.path == str(
        agglomerate_path
    )


def test_symlink_layer(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile.hdf5"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    agglomerate_path = (tmp_path / "agglomerate_view_15").resolve()
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

    copy_dataset = Dataset(tmp_path / "test_copy", voxel_size=(10, 10, 10))
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


def test_upload_fail(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()
    seg_layer.attachments.add_mesh(
        dataset.path / "seg" / "meshfile",
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    with pytest.raises(NotImplementedError):
        dataset.upload()


def test_unique_attachment_names(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "test_attachments", voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

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


def test_remote_dataset() -> None:
    dataset = Dataset(UPath("memory://test_attachments"), voxel_size=(10, 10, 10))
    seg_layer = dataset.add_layer(
        "seg",
        SEGMENTATION_CATEGORY,
        data_format="zarr3",
        bounding_box=BoundingBox((0, 0, 0), (16, 16, 16)),
    ).as_segmentation_layer()

    mesh_path = dataset.path / "seg" / "meshes" / "meshfile"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text("test")

    seg_layer.attachments.add_mesh(
        mesh_path,
        name="meshfile",
        data_format=AttachmentDataFormat.Zarr3,
    )

    assert seg_layer.attachments.meshes[0]._properties.path == "seg/meshes/meshfile"

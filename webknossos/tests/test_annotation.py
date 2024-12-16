import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import webknossos as wk
from webknossos.dataset import DataFormat
from webknossos.geometry import BoundingBox, Vec3Int

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


def test_annotation_from_wkw_zip_file() -> None:
    annotation = wk.Annotation.load(
        TESTDATA_DIR
        / "annotations"
        / "l4dense_motta_et_al_demo_v2__explorational__4a6356.zip"
    )

    assert annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert annotation.organization_id == "scalable_minds"
    assert annotation.owner_name == "Philipp Otto"
    assert annotation.annotation_id == "61c20205010000cc004a6356"
    assert "timestamp" in annotation.metadata
    assert len(list(annotation.get_volume_layer_names())) == 1
    assert len(list(annotation.skeleton.flattened_trees())) == 1

    annotation.save(TESTOUTPUT_DIR / "test_dummy.zip")
    copied_annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy.zip")

    assert copied_annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert copied_annotation.organization_id == "scalable_minds"
    assert copied_annotation.owner_name == "Philipp Otto"
    assert copied_annotation.annotation_id == "61c20205010000cc004a6356"
    assert "timestamp" in copied_annotation.metadata
    assert len(list(copied_annotation.get_volume_layer_names())) == 1
    assert len(list(copied_annotation.skeleton.flattened_trees())) == 1

    copied_annotation.add_volume_layer(name="new_volume_layer")
    assert len(list(copied_annotation.get_volume_layer_names())) == 2
    copied_annotation.delete_volume_layer(volume_layer_name="new_volume_layer")
    assert len(list(copied_annotation.get_volume_layer_names())) == 1

    with annotation.temporary_volume_layer_copy() as volume_layer:
        input_annotation_mag = volume_layer.get_finest_mag()
        voxel_id = input_annotation_mag.read(
            absolute_offset=Vec3Int(2830, 4356, 1792), size=Vec3Int.full(1)
        )

        assert voxel_id == 2504698


def test_annotation_from_zarr3_zip_file() -> None:
    annotation = wk.Annotation.load(
        TESTDATA_DIR / "annotations" / "l4_sample__explorational__suser__94b271.zip"
    )

    with annotation.temporary_volume_layer_copy() as volume_layer:
        assert volume_layer.data_format == DataFormat.Zarr3
        assert volume_layer.bounding_box == BoundingBox(
            (3072, 3072, 512), (1024, 1024, 1024)
        )
        input_annotation_mag = volume_layer.get_mag("2-2-1")
        voxel_id = input_annotation_mag.read(
            absolute_offset=Vec3Int(3630, 3502, 1024), size=Vec3Int(2, 2, 1)
        )

        assert np.array_equiv(voxel_id, 1)


def test_annotation_from_nml_file() -> None:
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_annotation_snapshot.nml"

    annotation = wk.Annotation.load(snapshot_path)

    assert annotation.dataset_name == "My Dataset"
    assert annotation.organization_id is None
    assert len(list(annotation.skeleton.flattened_trees())) == 3

    annotation.save(TESTOUTPUT_DIR / "test_dummy.zip")
    copied_annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy.zip")
    assert copied_annotation.dataset_name == "My Dataset"
    assert copied_annotation.organization_id is None
    assert len(list(copied_annotation.skeleton.flattened_trees())) == 3


def test_annotation_from_file_with_multi_volume() -> None:
    annotation = wk.Annotation.load(
        TESTDATA_DIR / "annotations" / "multi_volume_example_CREMI.zip"
    )

    volume_names = sorted(annotation.get_volume_layer_names())

    assert volume_names == ["Volume", "Volume_2"]

    # Read from first layer
    with annotation.temporary_volume_layer_copy(
        volume_layer_name=volume_names[0]
    ) as layer:
        read_voxel = layer.get_finest_mag().read(
            absolute_offset=(590, 512, 16),
            size=(1, 1, 1),
        )
        assert (
            read_voxel == 7718
        ), f"Expected to see voxel id 7718, but saw {read_voxel} instead."

        read_voxel = layer.get_finest_mag().read(
            absolute_offset=(490, 512, 16),
            size=(1, 1, 1),
        )
        # When viewing the annotation online, this segment id will be 284.
        # However, this is fallback data which is not included in this annotation.
        # Therefore, we expect to read a 0 here.
        assert (
            read_voxel == 0
        ), f"Expected to see voxel id 0, but saw {read_voxel} instead."

    # Read from second layer
    with annotation.temporary_volume_layer_copy(
        volume_layer_name=volume_names[1]
    ) as layer:
        read_voxel = layer.get_finest_mag().read(
            absolute_offset=(590, 512, 16),
            size=(1, 1, 1),
        )
        assert (
            read_voxel == 1
        ), f"Expected to see voxel id 1, but saw {read_voxel} instead."

        read_voxel = layer.get_finest_mag().read(
            absolute_offset=(490, 512, 16),
            size=(1, 1, 1),
        )
        assert (
            read_voxel == 0
        ), f"Expected to see voxel id 0, but saw {read_voxel} instead."

    # Reading from not-existing layer should raise an error
    with pytest.raises(AssertionError):
        with annotation.temporary_volume_layer_copy(
            volume_layer_name="not existing name"
        ) as layer:
            pass


@pytest.mark.use_proxay
def test_annotation_upload_download_roundtrip() -> None:
    path = TESTDATA_DIR / "annotations" / "l4_sample__explorational__suser__94b271.zip"
    annotation_from_file = wk.Annotation.load(path)
    annotation_from_file.organization_id = "Organization_X"
    test_token = os.getenv("WK_TOKEN")
    with wk.webknossos_context("http://localhost:9000", test_token):
        url = annotation_from_file.upload()
        annotation = wk.Annotation.download(url)
    assert annotation.dataset_name == "l4_sample"
    assert len(list(annotation.skeleton.flattened_trees())) == 1

    mag = wk.Mag("16-16-4")
    node_bbox = wk.BoundingBox.from_points(
        next(annotation.skeleton.flattened_trees()).get_node_positions()
    ).align_with_mag(mag, ceil=True)
    with wk.webknossos_context("http://localhost:9000", test_token):
        ds = annotation.get_remote_annotation_dataset()

    mag_view = ds.layers["Volume"].get_mag(mag)
    annotated_data = mag_view.read(absolute_bounding_box=node_bbox)
    assert annotated_data.size > 10
    # assert (annotated_data == 1).all()
    assert mag_view.read(absolute_offset=(0, 0, 0), size=(16, 16, 4))[0, 0, 0, 0] == 0
    assert (
        mag_view.read(absolute_offset=(3600, 3488, 1024), size=(16, 16, 4))[0, 0, 0, 0]
        == 1
    )
    segment_info = annotation.get_volume_layer_segments("Volume")[1]
    assert segment_info.anchor_position == (3395, 3761, 1024)
    segment_info.name = "Test Segment"
    segment_info.color = (1, 0, 0, 1)

    annotation.save(TESTOUTPUT_DIR / "test_dummy_downloaded.zip")
    annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy_downloaded.zip")
    assert annotation.dataset_name == "l4_sample"
    assert len(list(annotation.skeleton.flattened_trees())) == 1
    segment_info = annotation.get_volume_layer_segments("Volume")[1]
    assert segment_info.anchor_position == (3395, 3761, 1024)
    assert segment_info.name == "Test Segment"
    assert segment_info.color == (1, 0, 0, 1)


def test_reading_bounding_boxes() -> None:
    def check_properties(annotation: wk.Annotation) -> None:
        assert len(annotation.user_bounding_boxes) == 2
        assert annotation.user_bounding_boxes[0].topleft.x == 2371
        assert annotation.user_bounding_boxes[0].name == "Bounding box 1"
        assert annotation.user_bounding_boxes[0].is_visible

        assert annotation.user_bounding_boxes[1] == BoundingBox(
            (371, 4063, 1676), (891, 579, 232)
        )
        assert annotation.user_bounding_boxes[1].name == "Bounding box 2"
        assert not annotation.user_bounding_boxes[1].is_visible
        assert annotation.user_bounding_boxes[1].color == (
            0.2705882489681244,
            0.6470588445663452,
            0.19607843458652496,
            1.0,
        )

    # Check loading checked-in file
    input_path = TESTDATA_DIR / "annotations" / "bounding-boxes-example.zip"
    annotation = wk.Annotation.load(input_path)
    check_properties(annotation)

    # Check exporting and re-reading checked-in file (roundtrip)
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        output_path = Path(tmp_dir) / "serialized.zip"
        annotation.save(output_path)

        annotation_deserialized = wk.Annotation.load(output_path)
        check_properties(annotation_deserialized)


@pytest.mark.use_proxay
def test_bounding_box_roundtrip() -> None:
    ds = wk.Dataset.open_remote("l4_sample")

    annotation_before = wk.Annotation(
        name="test_bounding_box_roundtrip",
        dataset_name=ds.name,
        voxel_size=ds.voxel_size,
    )
    group = annotation_before.skeleton.add_group("a group")
    tree = group.add_tree("a tree")
    tree.add_node(position=(0, 0, 0), comment="node 1")

    annotation_before.user_bounding_boxes = [
        wk.BoundingBox((1024, 512, 128), (64, 64, 64))
    ]
    color = (0.5, 0, 0.2, 1)
    annotation_before.task_bounding_box = (
        wk.BoundingBox((10, 10, 10), (5, 5, 5)).with_name("task_bbox").with_color(color)
    )

    annotation_url = annotation_before.upload()
    annotation_after = wk.Annotation.download(annotation_url)

    # task bounding box is appended to user bounding boxes when uploading a normal annotation:
    assert (
        annotation_after.user_bounding_boxes
        == annotation_before.user_bounding_boxes + [annotation_before.task_bounding_box]
    )


def test_empty_volume_annotation() -> None:
    a = wk.Annotation.load(TESTDATA_DIR / "annotations" / "empty_volume_annotation.zip")
    with a.temporary_volume_layer_copy() as layer:
        assert layer.bounding_box == wk.BoundingBox.empty()
        assert layer.largest_segment_id == 0
        assert set(layer.mags) == set(
            [
                wk.Mag(1),
                wk.Mag((2, 2, 1)),
                wk.Mag((4, 4, 1)),
                wk.Mag((8, 8, 1)),
                wk.Mag((16, 16, 1)),
            ]
        )


@pytest.mark.parametrize(
    "nml_path",
    [
        TESTDATA_DIR / "annotations" / "nml_with_volumes.nml",
        TESTDATA_DIR / "annotations" / "nml_with_volumes.zip",
    ],
)
def test_nml_with_volumes(nml_path: Path) -> None:
    if nml_path.suffix == ".zip":
        with pytest.warns(UserWarning, match="location is not referenced in the NML"):
            a = wk.Annotation.load(nml_path)
    else:
        a = wk.Annotation.load(nml_path)
    segment_info = a.get_volume_layer_segments("segmentation")
    assert set(segment_info) == set([2504698])
    assert segment_info[2504698] == wk.SegmentInformation(
        name="test_segment", anchor_position=Vec3Int(3581, 3585, 1024), color=None
    )

import pytest

import webknossos as wk
from webknossos.geometry import Vec3Int

from .constants import TESTDATA_DIR, TESTOUTPUT_DIR


def test_annotation_from_zip_file() -> None:

    annotation = wk.Annotation.load(
        TESTDATA_DIR
        / "annotations"
        / "l4dense_motta_et_al_demo_v2__explorational__4a6356.zip"
    )

    assert annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert annotation.organization_id == "scalable_minds"
    assert annotation.username == "Philipp Otto"
    assert annotation.annotation_id == "61c20205010000cc004a6356"
    assert (
        "timestamp"
        in annotation.metadata  # pylint: disable=unsupported-membership-test
    )
    assert len(list(annotation.get_volume_layer_names())) == 1
    assert len(list(annotation.skeleton.flattened_graphs())) == 1

    annotation.save(TESTOUTPUT_DIR / "test_dummy.zip")
    copied_annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy.zip")

    assert copied_annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert copied_annotation.organization_id == "scalable_minds"
    assert copied_annotation.username == "Philipp Otto"
    assert copied_annotation.annotation_id == "61c20205010000cc004a6356"
    assert (
        "timestamp"
        in copied_annotation.metadata  # pylint: disable=unsupported-membership-test
    )
    assert len(list(copied_annotation.get_volume_layer_names())) == 1
    assert len(list(copied_annotation.skeleton.flattened_graphs())) == 1

    copied_annotation.add_volume_layer(name="new_volume_layer")
    assert len(list(copied_annotation.get_volume_layer_names())) == 2
    copied_annotation.delete_volume_layer(volume_layer_name="new_volume_layer")
    assert len(list(copied_annotation.get_volume_layer_names())) == 1

    with annotation.temporary_volume_layer_copy() as volume_layer:
        input_annotation_mag = volume_layer.get_best_mag()
        voxel_id = input_annotation_mag.read(
            absolute_offset=Vec3Int(2830, 4356, 1792), size=Vec3Int.full(1)
        )

        assert voxel_id == 2504698


def test_annotation_from_nml_file() -> None:
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_annotation_snapshot.nml"

    annotation = wk.Annotation.load(snapshot_path)

    assert annotation.dataset_name == "My Dataset"
    assert annotation.organization_id is None
    assert len(list(annotation.skeleton.flattened_graphs())) == 3

    annotation.save(TESTOUTPUT_DIR / "test_dummy.zip")
    copied_annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy.zip")
    assert copied_annotation.dataset_name == "My Dataset"
    assert copied_annotation.organization_id is None
    assert len(list(copied_annotation.skeleton.flattened_graphs())) == 3


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
        read_voxel = layer.get_best_mag().read(
            absolute_offset=(590, 512, 16),
            size=(1, 1, 1),
        )
        assert (
            read_voxel == 7718
        ), f"Expected to see voxel id 7718, but saw {read_voxel} instead."

        read_voxel = layer.get_best_mag().read(
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
        read_voxel = layer.get_best_mag().read(
            absolute_offset=(590, 512, 16),
            size=(1, 1, 1),
        )
        assert (
            read_voxel == 1
        ), f"Expected to see voxel id 1, but saw {read_voxel} instead."

        read_voxel = layer.get_best_mag().read(
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


def test_annotation_from_url() -> None:

    annotation = wk.Annotation.download(
        "https://webknossos.org/annotations/Explorational/61c20205010000cc004a6356"
    )
    assert annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert len(list(annotation.skeleton.flattened_graphs())) == 1

    annotation.save(TESTOUTPUT_DIR / "test_dummy_downloaded.zip")
    annotation = wk.Annotation.load(TESTOUTPUT_DIR / "test_dummy_downloaded.zip")
    assert annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert len(list(annotation.skeleton.flattened_graphs())) == 1

import sys
import os
import pytest

sys.path.insert(0, os.getcwd())

from pathlib import Path
from webknossos.geometry import Vec3Int
import webknossos as wk

TESTDATA_DIR = Path("testdata")


def test_annotation() -> None:

    annotation = wk.Annotation(
        TESTDATA_DIR
        / "annotations"
        / "l4dense_motta_et_al_demo_v2__explorational__4a6356.zip"
    )

    assert annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert len(list(annotation.skeleton.flattened_graphs())) == 1

    annotation.save_to_file("testoutput/test_dummy.zip")
    copied_annotation = wk.Annotation("testoutput/test_dummy.zip")
    assert copied_annotation.dataset_name == "l4dense_motta_et_al_demo_v2"
    assert len(list(copied_annotation.skeleton.flattened_graphs())) == 1

    with annotation.temporary_annotation_view() as volume_layer:
        input_annotation_mag = volume_layer.get_best_mag()
        voxel_id = input_annotation_mag.read(Vec3Int(2830, 4356, 1792), Vec3Int.full(1))

        assert voxel_id == 2504698

    with annotation._open_nml() as file_handle:
        skeleton_lines = file_handle.readlines()
        assert len(skeleton_lines) == 32

    # why does this fail with a permission error?
    # annotation = Annotation.download(
    #     "https://webknossos.org/annotations/Explorational/61c20205010000cc004a6356"
    # )
    # print(annotation)
    # print(annotation.dataset_name)
    # annotation.save_to_file(...)


if __name__ == "__main__":
    test_annotation()

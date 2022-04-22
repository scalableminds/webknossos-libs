import difflib
from os import PathLike
from pathlib import Path
from typing import List, Optional

import pytest

import webknossos as wk

from .constants import TESTDATA_DIR


def create_dummy_skeleton() -> wk.Skeleton:
    nml = wk.Skeleton(
        dataset_name="My Dataset",
        voxel_size=(11, 11, 25),
    )

    g = nml.add_tree(
        "A WkTree",
        color=(0.9988844805996959, 0.09300433970039235, 0.13373766240135082, 1.0),
    )

    n1 = g.add_node(
        position=(0, 1, 2),
        comment="A comment 1",
        is_branchpoint=True,
        inMag=0,
    )
    n2 = g.add_node(
        position=(3, 1, 2),
        comment="A comment 2",
    )
    n3 = g.add_node(
        position=(4, 1, 2),
        comment="A comment 3",
    )
    g.add_edge(n1, n2)
    g.add_edge(n1, n3)

    group = nml.add_group("Example Group")
    group.add_tree(
        "Tree in Group",
        color=(0.9340851110768926, 0.0037728487955197565, 0.6720369436532944, 1.0),
    ).add_node(position=(10, 3, 4))
    group.add_group("Nested Group").add_tree(
        "Tree in nested group",
        color=(0.45167026054501613, 0.20806732150346996, 0.7224589094338263, 1.0),
    )

    return nml


def test_immutability() -> None:
    skeleton = create_dummy_skeleton()

    with pytest.raises(AttributeError):
        skeleton.get_node_by_id(2).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        skeleton.get_tree_by_id(1).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        skeleton.get_group_by_id(5).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        skeleton.get_group_by_id(5).children = []  # type: ignore

    with pytest.raises(AttributeError):
        skeleton.get_group_by_id(5).children.append(skeleton.get_group_by_id(5))  # type: ignore


def test_skeleton_creation() -> None:
    skeleton = create_dummy_skeleton()

    trees = list(skeleton.flattened_trees())
    assert len(trees) == 3

    g1 = skeleton.get_tree_by_id(1)
    assert g1.number_of_nodes() == 3

    assert g1.get_node_by_id(2).comment == "A comment 1"
    assert g1.get_node_by_id(2).is_branchpoint
    assert g1.get_node_by_id(3).position == (3, 1, 2)

    groups = list(skeleton.flattened_groups())
    assert len(groups) == 2
    grand_children = [
        grand_child
        for grand_child in groups[0].children
        if isinstance(grand_child, wk.Tree)
    ]
    assert len(grand_children) == 1
    assert grand_children[0].group == groups[0]


def diff_lines(lines_a: List[str], lines_b: List[str]) -> List[str]:
    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile="a",
            tofile="b",
        )
    )
    return diff


def diff_files(path_a: PathLike, path_b: PathLike) -> None:
    with open(path_a, "r", encoding="utf-8") as file_a:
        with open(path_b, "r", encoding="utf-8") as file_b:
            diff = diff_lines(file_a.readlines(), file_b.readlines())
            assert (
                len(diff) == 0
            ), f"Files {path_a} and {path_b} are not equal:\n{''.join(diff)}"


def test_export_to_nml(tmp_path: Path) -> None:
    nml = create_dummy_skeleton()
    output_path = tmp_path / "out.nml"
    nml.save(output_path)

    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"

    diff_files(output_path, snapshot_path)


def test_import_from_nml() -> None:
    nml = create_dummy_skeleton()
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"
    loaded_nml = wk.Skeleton.load(snapshot_path)

    assert (
        nml == loaded_nml
    ), "NML created by create_dummy_skeleton() should equal NML loaded from disk."


def test_simple_initialization_and_representations(tmp_path: Path) -> None:
    nml = wk.Skeleton(dataset_name="ds_name", voxel_size=(0.5, 0.5, 0.5))
    nml_path = tmp_path / "my_skeleton.nml"
    EXPECTED_NML = """<?xml version="1.0" encoding="utf-8"?>
<things>
  <parameters>
    <experiment name="ds_name" />
    <scale x="0.5" y="0.5" z="0.5" />
  </parameters>
  <branchpoints />
  <comments />
  <groups />
</things>
"""
    nml.save(nml_path)
    with open(nml_path, "r", encoding="utf-8") as f:
        diff = diff_lines(f.readlines(), EXPECTED_NML.splitlines(keepends=True))
        assert (
            len(diff) == 0
        ), f"Written nml does not look as expected:\n{''.join(diff)}"
    assert nml == wk.Skeleton.load(nml_path)
    assert str(nml) == (
        "Skeleton(_child_groups=<No child groups>, _child_trees=<No child trees>, voxel_size=(0.5, 0.5, 0.5), dataset_name='ds_name', organization_id=None, description=None)"
    )

    my_group = nml.add_group("my_group")
    my_group.add_tree("my_tree", color=(0.1, 0.2, 0.3), _enforced_id=9).add_node(
        (2, 4, 6)
    )
    my_group.add_tree("my_other_tree", color=(0.1, 0.2, 0.3))
    nml.add_tree("top_level_tree", color=(0.1, 0.2, 0.3))

    EXPECTED_EXTENDED_NML = """<?xml version="1.0" encoding="utf-8"?>
<things>
  <parameters>
    <experiment name="ds_name" />
    <scale x="0.5" y="0.5" z="0.5" />
  </parameters>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" groupId="1" id="3" name="my_other_tree">
    <nodes />
    <edges />
  </thing>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" id="4" name="top_level_tree">
    <nodes />
    <edges />
  </thing>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" groupId="1" id="9" name="my_tree">
    <nodes>
      <node id="2" x="2.0" y="4.0" z="6.0" />
    </nodes>
    <edges />
  </thing>
  <branchpoints />
  <comments />
  <groups>
    <group id="1" name="my_group" />
  </groups>
</things>
"""
    nml.save(nml_path)
    with open(nml_path, "r", encoding="utf-8") as f:
        diff = diff_lines(
            f.readlines(), EXPECTED_EXTENDED_NML.splitlines(keepends=True)
        )
        assert (
            len(diff) == 0
        ), f"Written nml does not look as expected:\n{''.join(diff)}"
    assert nml == wk.Skeleton.load(nml_path)
    assert str(nml) == (
        "Skeleton(_child_groups=<1 child group>, _child_trees=<1 child tree>, voxel_size=(0.5, 0.5, 0.5), dataset_name='ds_name', organization_id=None, description=None)"
    )
    assert (
        str(my_group)
        == "Group(_id=1, name='my_group', _child_groups=<No child groups>, _child_trees=<2 child trees>)"
    )
    assert str(nml.get_tree_by_id(9)) == "Tree named 'my_tree' with 1 nodes and 0 edges"


def test_import_export_round_trip(tmp_path: Path) -> None:
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"
    export_path = tmp_path / "exported_in.nml"
    nml = wk.Skeleton.load(snapshot_path)

    g6 = nml.get_tree_by_id(6)
    assert g6.name == "Tree in Group"
    assert g6.get_node_by_id(7).position == (10.0, 3.0, 4.0)

    nml.save(export_path)
    diff_files(snapshot_path, export_path)


@pytest.mark.parametrize("layer_name", [None, "my_layer"])
def test_volume_dump_round_trip(tmp_path: Path, layer_name: Optional[str]) -> None:
    import xml.etree.ElementTree as ET

    from loxun import XmlWriter

    from webknossos._nml import Volume

    export_path = tmp_path / "volume_dump.xml"
    volume_in = Volume(
        id=0,
        location="data_Volume.zip",
        fallback_layer="my_very_important_layer",
        name=layer_name,
    )
    volume_out = None

    with open(export_path, "wb") as f:
        with XmlWriter(f) as xf:
            volume_in._dump(xf)

    with open(export_path, "rb") as f:
        tree = ET.parse(export_path)
        volume_out = Volume._parse(next(tree.iter()))

    assert volume_in == volume_out


def test_load_nml(tmp_path: Path) -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    output_path = tmp_path / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)
    skeleton_a.save(output_path)
    assert skeleton_a == wk.Skeleton.load(output_path)

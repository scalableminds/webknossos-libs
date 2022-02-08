import difflib
from os import PathLike
from pathlib import Path
from typing import List

import pytest

import webknossos as wk

from .constants import TESTDATA_DIR


def create_dummy_skeleton() -> wk.Skeleton:
    nml = wk.Skeleton(
        name="My NML",
        scale=(11, 11, 25),
        offset=(1, 1, 1),
        time=1337,
        edit_position=(3, 6, 0),
        edit_rotation=(4, 2, 0),
        zoom_level=100,
    )

    g = nml.add_graph(
        "A WkGraph",
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
    group.add_graph(
        "Graph in Group",
        color=(0.9340851110768926, 0.0037728487955197565, 0.6720369436532944, 1.0),
    ).add_node(position=(10, 3, 4))
    group.add_group("Nested Group").add_graph(
        "Graph in nested group",
        color=(0.45167026054501613, 0.20806732150346996, 0.7224589094338263, 1.0),
    )

    return nml


def test_immutability() -> None:
    nml = create_dummy_skeleton()

    with pytest.raises(AttributeError):
        nml.get_node_by_id(2).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        nml.get_graph_by_id(1).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        nml.get_group_by_id(5).id = 999  # type: ignore

    with pytest.raises(AttributeError):
        nml.get_group_by_id(5).children = []  # type: ignore

    with pytest.raises(AttributeError):
        nml.get_group_by_id(5).children.append(nml.get_group_by_id(5))  # type: ignore


def test_skeleton_creation() -> None:
    nml = create_dummy_skeleton()
    assert nml.time == 1337

    graphs = list(nml.flattened_graphs())
    assert len(graphs) == 3

    g1 = nml.get_graph_by_id(1)
    assert g1.number_of_nodes() == 3

    assert g1.get_node_by_id(2).comment == "A comment 1"
    assert g1.get_node_by_id(2).is_branchpoint
    assert g1.get_node_by_id(3).position == (3, 1, 2)

    groups = list(nml.flattened_groups())
    assert len(groups) == 2
    grand_children = [
        grand_child
        for grand_child in groups[0].children
        if isinstance(grand_child, wk.Graph)
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

    snapshot_path = TESTDATA_DIR / "nmls" / "generated_snapshot.nml"

    diff_files(output_path, snapshot_path)


def test_simple_initialization_and_representations(tmp_path: Path) -> None:
    nml = wk.Skeleton(name="my_skeleton", scale=(0.5, 0.5, 0.5), time=12345)
    nml_path = tmp_path / "my_skeleton.nml"
    EXPECTED_NML = """<?xml version="1.0" encoding="utf-8"?>
<things>
  <parameters>
    <experiment name="my_skeleton" />
    <scale x="0.5" y="0.5" z="0.5" />
    <time ms="12345" />
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
        "Skeleton(name='my_skeleton', _children=<No children>, scale=(0.5, 0.5, 0.5), offset=None, time=12345, "
        + "edit_position=None, edit_rotation=None, zoom_level=None, task_bounding_box=None, user_bounding_boxes=None)"
    )

    my_group = nml.add_group("my_group")
    my_group.add_graph("my_tree", color=(0.1, 0.2, 0.3), _enforced_id=9).add_node(
        (2, 4, 6)
    )
    my_group.add_graph("my_other_tree", color=(0.1, 0.2, 0.3))
    nml.add_graph("top_level_tree", color=(0.1, 0.2, 0.3))

    EXPECTED_EXTENDED_NML = """<?xml version="1.0" encoding="utf-8"?>
<things>
  <parameters>
    <experiment name="my_skeleton" />
    <scale x="0.5" y="0.5" z="0.5" />
    <time ms="12345" />
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
        "Skeleton(name='my_skeleton', _children=<2 children>, scale=(0.5, 0.5, 0.5), offset=None, time=12345, "
        + "edit_position=None, edit_rotation=None, zoom_level=None, task_bounding_box=None, user_bounding_boxes=None)"
    )
    assert str(my_group) == "Group(_id=1, name='my_group', _children=<2 children>)"
    assert (
        str(nml.get_graph_by_id(9)) == "Graph named 'my_tree' with 1 nodes and 0 edges"
    )


def test_import_export_round_trip(tmp_path: Path) -> None:
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_snapshot.nml"
    export_path = tmp_path / "exported_in.nml"
    nml = wk.Skeleton.load(snapshot_path)
    assert nml.time == 1337

    g6 = nml.get_graph_by_id(6)
    assert g6.name == "Graph in Group"
    assert g6.get_node_by_id(7).position == (10.0, 3.0, 4.0)

    nml.save(export_path)
    diff_files(snapshot_path, export_path)


def test_volume_dump_round_trip(tmp_path: Path) -> None:
    import xml.etree.ElementTree as ET

    from loxun import XmlWriter

    from webknossos.skeleton.nml import Volume, __dump_volume, __parse_volume

    export_path = tmp_path / "volume_dump.xml"
    volume_in = Volume(
        id=0, location="data_Volume.zip", fallback_layer="my_very_important_layer"
    )
    volume_out = None

    with open(export_path, "wb") as f:
        with XmlWriter(f) as xf:
            __dump_volume(xf, volume_in)

    with open(export_path, "rb") as f:
        tree = ET.parse(export_path)
        volume_out = __parse_volume(next(tree.iter()))

    assert volume_in == volume_out

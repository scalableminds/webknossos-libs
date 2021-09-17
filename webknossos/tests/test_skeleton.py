import difflib
import tempfile

import numpy as np
import pytest

import webknossos.skeleton as skeleton
from webknossos.examples.skeleton_synapse_candidates import find_synapse_candidates


def create_dummy_skeleton() -> skeleton.Skeleton:
    nml = skeleton.Skeleton(
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
    assert len(g1.get_nodes()) == 3

    assert g1.get_node_by_id(2).comment == "A comment 1"
    assert g1.get_node_by_id(2).is_branchpoint
    assert g1.get_node_by_id(3).position == (3, 1, 2)

    groups = list(nml.flattened_groups())
    assert len(groups) == 2
    grand_child = next(groups[0].children)
    assert isinstance(grand_child, skeleton.Graph)
    assert grand_child.group_id == groups[0].id


def diff_files(path_a: str, path_b: str) -> None:
    with open(path_a, "r") as file_a:
        with open(path_b, "r") as file_b:
            diff = list(
                difflib.unified_diff(
                    file_a.readlines(),
                    file_b.readlines(),
                    fromfile="file_a",
                    tofile="file_b",
                )
            )
            if len(diff) > 0:
                print("####################")
                for line in diff:
                    print(line)
                print("####################")
            assert len(diff) == 0, f"Files {path_a} and {path_b} are not equal: {diff}"


def test_export_to_nml() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:

        nml = create_dummy_skeleton()
        output_path = f"{temp_dir}/out.nml"
        nml.write(output_path)

        snapshot_path = "../testdata/nmls/generated_snapshot.nml"

        diff_files(output_path, snapshot_path)


def test_import_export_round_trip() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_path = "../testdata/nmls/generated_snapshot.nml"
        export_path = f"{temp_dir}/exported_in.nml"
        nml = skeleton.Skeleton.from_path(snapshot_path)
        assert nml.time == 1337

        g6 = nml.get_graph_by_id(6)
        assert g6.name == "Graph in Group"
        assert g6.get_node_by_id(7).position == (10.0, 3.0, 4.0)

        nml.write(export_path)
        diff_files(snapshot_path, export_path)


def test_code_example() -> None:

    nml, synapse_parent_group = find_synapse_candidates()

    assert synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."

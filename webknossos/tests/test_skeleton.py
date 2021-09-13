# import pytest

# with pytest.raises(AssertionError):
import webknossos.skeleton as skeleton
import tempfile
import difflib
from itertools import combinations
from typing import Generator, Tuple
import numpy as np


def pairs_within_distance(
    pos_a: np.ndarray, pos_b: np.ndarray, max_distance: float
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    from scipy.spatial import cKDTree

    pos_a_kdtree = cKDTree(pos_a)
    pos_b_kdtree = cKDTree(pos_b)
    indexes = pos_a_kdtree.query_ball_tree(pos_b_kdtree, max_distance)
    for i in range(len(indexes)):
        for j in indexes[i]:
            yield (pos_a[i], pos_b[j])


def create_dummy_skeleton() -> skeleton.NML:
    nml = skeleton.NML(
        name="My NML",
        scale=(11, 11, 25),
        offset=(1, 1, 1),
        time=1337,
        editPosition=(3, 6, 0),
        editRotation=(4, 2, 0),
        zoomLevel=100,
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
        nml = skeleton.NML.from_path(snapshot_path)
        assert nml.time == 1337

        g6 = nml.get_graph_by_id(6)
        assert g6.name == "Graph in Group"
        assert g6.get_node_by_id(7).position == (10.0, 3.0, 4.0)

        nml.write(export_path)
        diff_files(snapshot_path, export_path)


def test_code_example() -> None:

    nml = skeleton.open_nml("../testdata/nmls/nml_with_small_distance_nodes.nml")

    synapse_candidate_max_distance = 0.5  # in nm

    input_graphs = nml.flattened_graphs()
    synapse_parent_group = nml.add_group("all synapse candidates")

    for tree_a, tree_b in combinations(input_graphs, 2):
        pos_a = tree_a.get_node_positions() * nml.scale
        pos_b = tree_b.get_node_positions() * nml.scale

        synapse_graph = synapse_parent_group.add_graph(
            f"synapse candidates ({tree_a.name}-{tree_b.name})"
        )

        for partner_a, partner_b in pairs_within_distance(
            pos_a, pos_b, synapse_candidate_max_distance
        ):
            synapse_graph.add_node(
                position=(partner_a + partner_b) / nml.scale / 2,
                comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
            )

    assert synapse_parent_group.get_total_node_count() == 57
    ids = [g.id for g in nml.flattened_graphs()]
    id_set = set(ids)
    assert len(ids) == len(id_set), "Graph IDs are not unique."

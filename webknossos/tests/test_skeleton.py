# import pytest

# with pytest.raises(AssertionError):
import webknossos.skeleton as skeleton
import tempfile
import difflib


def create_dummy_skeleton():
    nml = skeleton.NML(name="My NML", scale=(11, 11, 25))
    g = nml.add_graph(
        "A WkGraph",
        color=[0.9988844805996959, 0.09300433970039235, 0.13373766240135082, 1.0],
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
        color=[0.9340851110768926, 0.0037728487955197565, 0.6720369436532944, 1.0],
    ).add_node(position=[10, 3, 4])
    group.add_group("Nested Group").add_graph(
        "Graph in nested group",
        color=[0.45167026054501613, 0.20806732150346996, 0.7224589094338263, 1.0],
    )

    return nml


def test_skeleton_creation():
    nml = create_dummy_skeleton()

    graphs = list(nml.flattened_graphs())
    assert len(graphs) == 3

    g1 = nml.get_graph_by_id(1)
    assert len(g1.get_nodes()) == 3

    assert g1.get_node_by_id(2).comment == "A comment 1"
    assert g1.get_node_by_id(2).is_branchpoint
    assert g1.get_node_by_id(3).position == (3, 1, 2)

    groups = list(nml.flattened_groups())
    assert len(groups) == 2


def diff_files(path_a, path_b):
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
            for line in diff:
                print(diff)
            assert len(diff) == 0, f"File are not equal: {diff}"


def test_export_to_nml():
    with tempfile.TemporaryDirectory() as temp_dir:

        nml = create_dummy_skeleton()
        output_path = f"{temp_dir}/out.nml"
        nml.write(output_path)

        snapshot_path = "../testdata/nmls/generated_snapshot.nml"

        diff_files(output_path, snapshot_path)


def test_import_export_round_trip():
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_path = "../testdata/nmls/generated_snapshot.nml"
        # export_path = f"{temp_dir}/exported_in.nml"
        export_path = f"exported_in.nml"
        nml = skeleton.NML.from_path(snapshot_path)
        nml.write(export_path)

        # diff_files(snapshot_path, export_path)

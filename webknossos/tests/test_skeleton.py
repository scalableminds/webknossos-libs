import difflib

import networkx as nx
import pytest
from upath import UPath

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


def create_dummy_nx_graph() -> nx.Graph:
    nx_graph = nx.Graph()
    nx_graph.add_node(1, position=(0, 1, 2), comment="node 1 nx")
    nx_graph.add_node(2, position=(3, 1, 2), comment="node 2 nx")
    nx_graph.add_edge(1, 2)

    return nx_graph


def test_doc_example() -> None:
    from webknossos import Annotation

    annotation = Annotation(
        name="my_annotation",
        dataset_name="my_dataset",
        voxel_size=(11, 11, 24),
    )
    group = annotation.skeleton.add_group("a group")
    tree = group.add_tree("a tree")
    node_1 = tree.add_node(position=(0, 0, 0), comment="node 1")
    node_2 = tree.add_node(position=(100, 100, 100), comment="node 2")

    tree.add_edge(node_1, node_2)


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


def test_add_nx_graph() -> None:
    skeleton = create_dummy_skeleton()
    node_count = skeleton.get_total_node_count()
    tree_count = len(list(skeleton.flattened_trees()))
    group_count = len(list(skeleton.flattened_groups()))
    max_node_id = skeleton.get_max_node_id()

    nx_graph = create_dummy_nx_graph()
    skeleton.add_nx_graphs(
        {"first_group": [nx_graph, nx_graph], "second_group": [nx_graph]}
    )

    # check number of groups, nodes and trees
    assert len(list(skeleton.flattened_groups())) == group_count + 2
    assert skeleton.get_total_node_count() == node_count + 6
    assert len(list(skeleton.flattened_trees())) == tree_count + 3

    # check group names
    for group in skeleton.flattened_groups():
        assert group.name in [
            "first_group",
            "second_group",
            "Example Group",
            "Nested Group",
        ]

    # check node attributes
    max_node_id = skeleton.get_max_node_id()
    assert skeleton.get_node_by_id(max_node_id).comment == "node 2 nx"
    assert skeleton.get_node_by_id(max_node_id).position == (3, 1, 2)
    assert skeleton.get_node_by_id(max_node_id - 1).comment == "node 1 nx"
    assert skeleton.get_node_by_id(max_node_id - 1).position == (0, 1, 2)

    # check if edge was added
    for edge in skeleton.get_tree_by_id(max_node_id - 2).edges:
        assert (edge[0].id, edge[1].id) == (max_node_id - 1, max_node_id)


def test_nml_generation(tmp_upath: UPath) -> None:
    OLD_NML_PATH = TESTDATA_DIR / "nmls" / "generate_nml_snapshot.nml"

    tree1 = create_dummy_nx_graph()
    tree2 = create_dummy_nx_graph()
    tree2.add_node(3, position=(3, 3, 3), comment="node 3 nx")

    tree_dict = {"first_group": [tree1], "second_group": [tree2]}

    # old_nml was generated with the old wknml library as follows:
    # params_wknml = {"name": "MyDataset", "scale": (1, 1, 1), "zoomLevel": 0.4}
    # old_nml = generate_nml(tree_dict=tree_dict, parameters=params_wknml)
    # with open(tmp_upath / "annotation_old.nml", "wb") as f:
    #     write_nml(f, old_nml)

    tree_dict = {"first_group": [tree1], "second_group": [tree2]}

    annotation = wk.Annotation(
        name="MyAnnotation",
        dataset_name="MyDataset",
        voxel_size=(1, 1, 1),
        zoom_level=0.4,
    )

    annotation.skeleton.add_nx_graphs(tree_dict)

    annotation.save(tmp_upath / "annotation_new.nml")

    old_skeleton = wk.Skeleton.load(OLD_NML_PATH)
    new_skeleton = wk.Skeleton.load(tmp_upath / "annotation_new.nml")

    for old_group, new_group in zip(
        old_skeleton.flattened_groups(), new_skeleton.flattened_groups()
    ):
        assert old_group.name == new_group.name
        for old_child, new_child in zip(old_group.children, new_group.children):
            if isinstance(old_child, wk.Tree) and isinstance(new_child, wk.Tree):
                for old_node, new_node in zip(old_child.nodes, new_child.nodes):
                    assert old_node.comment == new_node.comment
                    assert old_node.position == new_node.position
                    assert old_node.radius == new_node.radius
                for old_edge, new_edge in zip(old_child.edges, new_child.edges):
                    assert old_edge[0].position == new_edge[0].position
                    assert old_edge[1].position == new_edge[1].position


def diff_lines(lines_a: list[str], lines_b: list[str]) -> list[str]:
    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile="a",
            tofile="b",
        )
    )
    return diff


def diff_files(path_a: UPath, path_b: UPath) -> None:
    with path_a.open("r", encoding="utf-8") as file_a:
        with path_b.open("r", encoding="utf-8") as file_b:
            diff = diff_lines(file_a.readlines(), file_b.readlines())
            assert len(diff) == 0, (
                f"Files {path_a} and {path_b} are not equal:\n{''.join(diff)}"
            )


def test_export_to_nml(tmp_upath: UPath) -> None:
    nml = create_dummy_skeleton()
    output_path = tmp_upath / "out.nml"
    nml.save(output_path)

    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"

    diff_files(output_path, snapshot_path)


def test_import_from_nml() -> None:
    nml = create_dummy_skeleton()
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"
    loaded_nml = wk.Skeleton.load(snapshot_path)

    assert nml == loaded_nml, (
        "NML created by create_dummy_skeleton() should equal NML loaded from disk."
    )


def test_simple_initialization_and_representations(tmp_upath: UPath) -> None:
    nml = wk.Skeleton(dataset_name="ds_name", voxel_size=(0.5, 0.5, 0.5))
    nml_path = tmp_upath / "my_skeleton.nml"
    EXPECTED_NML = """<?xml version="1.0" encoding="utf-8"?>
<things>
  <parameters>
    <experiment name="ds_name" />
    <scale unit="nanometer" x="0.5" y="0.5" z="0.5" />
  </parameters>
  <branchpoints />
  <comments />
  <groups />
</things>
"""
    nml.save(nml_path)
    with nml_path.open(mode="r", encoding="utf-8") as f:
        diff = diff_lines(f.readlines(), EXPECTED_NML.splitlines(keepends=True))
        assert len(diff) == 0, (
            f"Written nml does not look as expected:\n{''.join(diff)}"
        )
    assert nml == wk.Skeleton.load(nml_path)
    assert str(nml) == (
        "Skeleton(_child_groups=<No child groups>, _child_trees=<No child trees>, _voxel_size=VoxelSize(factor=Vec3Float(0.5,0.5,0.5), unit=<LengthUnit.NANOMETER: 'nanometer'>), dataset_name='ds_name', dataset_id=None, organization_id=None, description=None)"
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
    <scale unit="nanometer" x="0.5" y="0.5" z="0.5" />
  </parameters>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" groupId="1" id="9" isVisible="true" name="my_tree">
    <nodes>
      <node id="10" x="2.0" y="4.0" z="6.0" />
    </nodes>
    <edges />
    <metadata />
  </thing>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" groupId="1" id="11" isVisible="true" name="my_other_tree">
    <nodes />
    <edges />
    <metadata />
  </thing>
  <thing color.a="1.0" color.b="0.3" color.g="0.2" color.r="0.1" id="12" isVisible="true" name="top_level_tree">
    <nodes />
    <edges />
    <metadata />
  </thing>
  <branchpoints />
  <comments />
  <groups>
    <group id="1" name="my_group" />
  </groups>
</things>
"""
    nml.save(nml_path)
    with nml_path.open("r", encoding="utf-8") as f:
        diff = diff_lines(
            f.readlines(), EXPECTED_EXTENDED_NML.splitlines(keepends=True)
        )
        assert len(diff) == 0, (
            f"Written nml does not look as expected:\n{''.join(diff)}"
        )
    assert nml == wk.Skeleton.load(nml_path)
    assert str(nml) == (
        "Skeleton(_child_groups=<1 child group>, _child_trees=<1 child tree>, _voxel_size=VoxelSize(factor=Vec3Float(0.5,0.5,0.5), unit=<LengthUnit.NANOMETER: 'nanometer'>), dataset_name='ds_name', dataset_id=None, organization_id=None, description=None)"
    )
    assert (
        str(my_group)
        == "Group(_id=1, name='my_group', _child_groups=<No child groups>, _child_trees=<2 child trees>)"
    )
    assert str(nml.get_tree_by_id(9)) == "Tree named 'my_tree' with 1 nodes and 0 edges"


def test_import_export_round_trip(tmp_upath: UPath) -> None:
    snapshot_path = TESTDATA_DIR / "nmls" / "generated_skeleton_snapshot.nml"
    export_path = tmp_upath / "exported_in.nml"
    nml = wk.Skeleton.load(snapshot_path)

    g6 = nml.get_tree_by_id(6)
    assert g6.name == "Tree in Group"
    assert g6.get_node_by_id(7).position == (10.0, 3.0, 4.0)

    nml.save(export_path)
    diff_files(snapshot_path, export_path)


@pytest.mark.parametrize("layer_name", [None, "my_layer"])
def test_volume_dump_round_trip(tmp_upath: UPath, layer_name: str | None) -> None:
    import xml.etree.ElementTree as ET

    from loxun import XmlWriter

    from webknossos._nml import Volume

    export_path = tmp_upath / "volume_dump.xml"
    volume_in = Volume(
        id=0,
        location="data_Volume.zip",
        fallback_layer="my_very_important_layer",
        name=layer_name,
        segments=[],
    )
    volume_out = None

    with export_path.open("wb") as f:
        with XmlWriter(f) as xf:
            volume_in._dump(xf)

    with export_path.open("rb") as f:
        tree = ET.parse(f)
        volume_out = Volume._parse(next(tree.iter()))

    assert volume_in == volume_out


def test_load_nml(tmp_upath: UPath) -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    output_path = tmp_upath / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)
    skeleton_a.save(output_path)
    assert skeleton_a == wk.Skeleton.load(output_path)


def test_load_nml_with_scale_unit() -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)
    assert skeleton_a.voxel_size == (4000.0, 4000.0, 35000.0)


def test_remove_tree(tmp_upath: UPath) -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    output_path = tmp_upath / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)

    # Check that tree exists
    tree = skeleton_a.get_tree_by_id(1)
    assert tree is not None
    assert tree in list(skeleton_a.children)

    # Check that tree doesn't exist anymore
    skeleton_a.remove_tree_by_id(1)
    with pytest.raises(ValueError):
        tree = skeleton_a.get_tree_by_id(1)

    assert tree not in list(skeleton_a.children)

    # Check that serialized skeleton doesn't contain
    # deleted tree
    skeleton_a.save(output_path)
    assert skeleton_a == wk.Skeleton.load(output_path)

    # Load original file and check that tree is still
    # there (should not have been removed on disk
    # automatically).
    skeleton_a = wk.Skeleton.load(input_path)
    assert tree in list(skeleton_a.children)


def test_add_tree_with_obj(tmp_upath: UPath) -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    output_path = tmp_upath / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)

    # Check that tree exists
    tree = skeleton_a.get_tree_by_id(1)

    skeleton_b = wk.Skeleton(
        voxel_size=skeleton_a.voxel_size, dataset_name=skeleton_a.dataset_name
    )
    skeleton_b.add_tree(tree)

    assert tree is not skeleton_b.get_tree_by_id(1)
    assert tree == skeleton_b.get_tree_by_id(1)

    skeleton_b.save(output_path)


def test_add_tree_with_obj_and_properties(tmp_upath: UPath) -> None:
    input_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    output_path = tmp_upath / "test_a.nml"
    skeleton_a = wk.Skeleton.load(input_path)

    # Check that tree exists
    tree = skeleton_a.get_tree_by_id(1)

    skeleton_b = wk.Skeleton(
        voxel_size=skeleton_a.voxel_size, dataset_name=skeleton_a.dataset_name
    )
    new_tree = skeleton_b.add_tree(tree, color=(1, 2, 3), _enforced_id=1337)

    assert new_tree is skeleton_b.get_tree_by_id(1337)
    assert new_tree is not tree
    assert new_tree != tree
    assert new_tree.color == (1, 2, 3, 1)

    skeleton_b.save(output_path)


def test_add_tree_with_group() -> None:
    annotation = wk.Annotation(
        name="my_annotation", dataset_name="my_dataset", voxel_size=(11, 11, 24)
    )
    group = annotation.skeleton.add_group("a group")
    tree = group.add_tree("a tree")

    skeleton_a = create_dummy_skeleton()

    skeleton_a.add_tree(tree)


def test_is_visible(tmp_upath: UPath) -> None:
    skeleton = wk.Skeleton(dataset_name="test", voxel_size=(1, 1, 1))

    # Default is visible
    tree_visible = skeleton.add_tree("visible_tree")
    assert tree_visible.is_visible

    # Explicit False
    tree_hidden = skeleton.add_tree("hidden_tree", is_visible=False)
    assert not tree_hidden.is_visible

    # Trees with different is_visible are not equal
    assert tree_visible != tree_hidden

    # NML round-trip preserves is_visible=False
    nml_path = tmp_upath / "is_visible_test.nml"
    skeleton.add_tree("visible_tree_2", is_visible=True)
    skeleton.save(nml_path)
    loaded = wk.Skeleton.load(nml_path)

    loaded_trees = {t.name: t for t in loaded.flattened_trees()}
    assert loaded_trees["visible_tree"].is_visible
    assert not loaded_trees["hidden_tree"].is_visible
    assert loaded_trees["visible_tree_2"].is_visible

    # Loading NML without isVisible attribute defaults to True
    nml_without_is_visible = tmp_upath / "no_is_visible.nml"
    nml_without_is_visible.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<things>\n"
        '  <parameters><experiment name="test" /><scale x="1.0" y="1.0" z="1.0" /></parameters>\n'
        '  <thing color.a="1.0" color.b="0.0" color.g="0.0" color.r="1.0" id="1" name="legacy_tree">\n'
        "    <nodes /><edges /><metadata />\n"
        "  </thing>\n"
        "  <branchpoints /><comments /><groups />\n"
        "</things>\n"
    )
    loaded_legacy = wk.Skeleton.load(nml_without_is_visible)
    assert loaded_legacy.get_tree_by_id(1).is_visible


def test_behave_like_nx_graph() -> None:
    s = wk.Skeleton((1, 1, 1), dataset_name="test_dataset")
    g = s.add_group("test")
    tree = g.add_tree("t")
    node1 = tree.add_node((1, 2, 3))
    node2 = tree.add_node((4, 5, 6))
    tree.add_edge(node1, node2)

    for node, node_data in tree.nodes(data=True):
        assert isinstance(node, wk.Node)
        assert isinstance(node_data, dict)
    assert len(tree.nodes) == 2

    # remove nodes with position (1,2,3)
    nodes_to_remove = []
    for node, node_data in tree.nodes(data=True):
        if node_data["position"].to_tuple() == (1, 2, 3):
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        tree.remove_node(node)
    assert len(tree.nodes) == 1


def _make_colliding_skeleton(prefix: str) -> wk.Skeleton:
    # Same tree, node and group ids in every skeleton created by this helper.
    skeleton = wk.Skeleton(dataset_name="test_dataset", voxel_size=(1, 1, 1))
    top_tree = skeleton.add_tree(
        f"{prefix}_top", color=(1, 0, 0, 1), metadata={"origin": prefix}
    )
    a = top_tree.add_node((1, 1, 1), comment=f"{prefix}_a", is_branchpoint=True)
    b = top_tree.add_node((2, 2, 2), radius=2.5)
    c = top_tree.add_node((3, 3, 3))
    top_tree.add_edge(a, b)
    top_tree.add_edge(b, c)

    group = skeleton.add_group(f"{prefix}_group")
    nested = group.add_group(f"{prefix}_nested")
    tree = nested.add_tree(f"{prefix}_nested_tree", is_visible=False)
    d = tree.add_node((4, 4, 4), comment=f"{prefix}_d")
    e = tree.add_node((5, 5, 5))
    tree.add_edge(d, e)
    return skeleton


def _all_ids(group: wk.Group) -> tuple[list[int], list[int], list[int]]:
    group_ids = [g.id for g in group.flattened_groups()]
    tree_ids = [t.id for t in group.flattened_trees()]
    node_ids = [n.id for t in group.flattened_trees() for n in t.nodes]
    return group_ids, tree_ids, node_ids


def test_add_skeleton_merges_colliding_skeletons(tmp_upath: UPath) -> None:
    sources = {"a": _make_colliding_skeleton("a"), "b": _make_colliding_skeleton("b")}
    assert _all_ids(sources["a"]) == _all_ids(sources["b"])

    target = wk.Skeleton(dataset_name="test_dataset", voxel_size=(1, 1, 1))
    for name, source in sources.items():
        target.add_group(f"file_{name}").add_skeleton(source)

    # sources are unchanged
    assert _all_ids(sources["a"]) == _all_ids(sources["b"])

    def check(skeleton: wk.Skeleton) -> None:
        group_ids, tree_ids, node_ids = _all_ids(skeleton)
        assert len(group_ids) == len(set(group_ids)) == 6
        assert len(tree_ids) == len(set(tree_ids)) == 4
        assert len(node_ids) == len(set(node_ids)) == 10

        top_groups = {g.name: g for g in skeleton.groups}
        assert set(top_groups) == {"file_a", "file_b"}
        for prefix in ("a", "b"):
            file_group = top_groups[f"file_{prefix}"]
            assert [t.name for t in file_group.trees] == [f"{prefix}_top"]
            (top_tree,) = file_group.trees
            assert top_tree.color == (1, 0, 0, 1)
            assert top_tree.metadata == {"origin": prefix}
            assert top_tree.number_of_nodes() == 3
            assert top_tree.number_of_edges() == 2
            nodes = {n.position.to_tuple(): n for n in top_tree.nodes}
            assert nodes[(1, 1, 1)].comment == f"{prefix}_a"
            assert nodes[(1, 1, 1)].is_branchpoint
            assert nodes[(2, 2, 2)].radius == 2.5
            assert top_tree.has_edge(nodes[(1, 1, 1)], nodes[(2, 2, 2)])
            assert top_tree.has_edge(nodes[(2, 2, 2)], nodes[(3, 3, 3)])
            assert not top_tree.has_edge(nodes[(1, 1, 1)], nodes[(3, 3, 3)])

            (group,) = file_group.groups
            assert group.name == f"{prefix}_group"
            (nested,) = group.groups
            assert nested.name == f"{prefix}_nested"
            (nested_tree,) = nested.trees
            assert nested_tree.name == f"{prefix}_nested_tree"
            assert not nested_tree.is_visible
            nested_nodes = {n.position.to_tuple(): n for n in nested_tree.nodes}
            assert nested_nodes[(4, 4, 4)].comment == f"{prefix}_d"
            assert nested_tree.has_edge(
                nested_nodes[(4, 4, 4)], nested_nodes[(5, 5, 5)]
            )

    check(target)

    path = tmp_upath / "merged.nml"
    target.save(path)
    check(wk.Skeleton.load(path))


def test_add_skeleton_to_annotation_loaded_from_nml() -> None:
    nml_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    annotation = wk.Annotation(
        name="merged", dataset_name="test_dataset", voxel_size=(1, 1, 1)
    )
    for _ in range(2):
        loaded = wk.Annotation.load(nml_path)
        annotation.skeleton.add_group("file").add_skeleton(loaded.skeleton)
    expected_nodes = 2 * loaded.skeleton.get_total_node_count()

    _, tree_ids, node_ids = _all_ids(annotation.skeleton)
    assert len(node_ids) == len(set(node_ids)) == expected_nodes
    assert len(tree_ids) == len(set(tree_ids))

    # fresh ids of later additions do not collide
    extra = annotation.skeleton.add_tree("extra")
    extra.add_node((0, 0, 0))
    _, _, node_ids = _all_ids(annotation.skeleton)
    assert len(node_ids) == len(set(node_ids))


def test_add_skeleton_into_itself_raises() -> None:
    skeleton = _make_colliding_skeleton("a")
    with pytest.raises(ValueError):
        skeleton.add_group("g").add_skeleton(skeleton)


def test_add_tree_copies_with_colliding_ids(tmp_upath: UPath) -> None:
    nml_path = TESTDATA_DIR / "nmls" / "test_a.nml"
    sources = [wk.Skeleton.load(nml_path), wk.Skeleton.load(nml_path)]
    assert _all_ids(sources[0]) == _all_ids(sources[1])
    source_trees = [
        (t.name, t.number_of_nodes(), t.number_of_edges())
        for t in sources[0].flattened_trees()
    ]

    target = wk.Skeleton(dataset_name="test_dataset", voxel_size=(1, 1, 1))
    for source in sources:
        for tree in list(source.flattened_trees()):
            target.add_tree(tree)

    def check(skeleton: wk.Skeleton) -> None:
        _, tree_ids, node_ids = _all_ids(skeleton)
        assert len(tree_ids) == len(set(tree_ids)) == 2 * len(source_trees)
        assert len(node_ids) == len(set(node_ids))
        assert sorted(
            (t.name, t.number_of_nodes(), t.number_of_edges())
            for t in skeleton.flattened_trees()
        ) == sorted(source_trees * 2)
        for tree in skeleton.flattened_trees():
            for u, v in tree.edges:
                assert tree.has_node(u.id) and tree.has_node(v.id)

    check(target)
    path = tmp_upath / "copied.nml"
    target.save(path)
    check(wk.Skeleton.load(path))

    # ids handed out later do not collide either
    target.add_tree("new").add_node((0, 0, 0))
    _, tree_ids, node_ids = _all_ids(target)
    assert len(tree_ids) == len(set(tree_ids))
    assert len(node_ids) == len(set(node_ids))


def test_enforced_ids_are_not_handed_out_again() -> None:
    skeleton = wk.Skeleton(dataset_name="test_dataset", voxel_size=(1, 1, 1))
    tree = skeleton.add_tree("a", _enforced_id=5)
    tree.add_node((0, 0, 0), _enforced_id=7)
    group = skeleton.add_group("g", _enforced_id=9)
    for _ in range(10):
        group.add_tree("b").add_node((1, 1, 1))
    group_ids, tree_ids, node_ids = _all_ids(skeleton)
    assert len(tree_ids) == len(set(tree_ids))
    assert len(node_ids) == len(set(node_ids))
    assert len(group_ids) == len(set(group_ids))


def test_add_tree_copies_keep_edges_with_colliding_ids(tmp_upath: UPath) -> None:
    sources = [_make_colliding_skeleton("a"), _make_colliding_skeleton("b")]
    target = wk.Skeleton(dataset_name="test_dataset", voxel_size=(1, 1, 1))
    for source in sources:
        for tree in list(source.flattened_trees()):
            target.add_tree(tree)

    # sources are unchanged
    assert _all_ids(sources[0]) == _all_ids(sources[1])

    def check(skeleton: wk.Skeleton) -> None:
        _, tree_ids, node_ids = _all_ids(skeleton)
        assert len(tree_ids) == len(set(tree_ids)) == 4
        assert len(node_ids) == len(set(node_ids)) == 10
        for tree in skeleton.flattened_trees():
            positions = {n.position.to_tuple(): n for n in tree.nodes}
            if tree.name.endswith("_top"):
                assert tree.number_of_edges() == 2
                assert tree.has_edge(positions[(1, 1, 1)], positions[(2, 2, 2)])
                assert tree.has_edge(positions[(2, 2, 2)], positions[(3, 3, 3)])
                assert positions[(1, 1, 1)].is_branchpoint
            else:
                assert tree.number_of_edges() == 1
                assert tree.has_edge(positions[(4, 4, 4)], positions[(5, 5, 5)])

    check(target)
    path = tmp_upath / "copied.nml"
    target.save(path)
    check(wk.Skeleton.load(path))

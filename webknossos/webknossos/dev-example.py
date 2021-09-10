import itertools
from itertools import combinations
import networkx as nx
from icecream import ic


def pairs_within_distance(pos_a, pos_b, max_distance):
    from scipy.spatial import cKDTree

    pos_a_kdtree = cKDTree(pos_a)
    pos_b_kdtree = cKDTree(pos_b)
    indexes = pos_a_kdtree.query_ball_tree(pos_b_kdtree, max_distance)
    for i in range(len(indexes)):
        for j in indexes[i]:
            yield (pos_a[i], pos_b[j])


def dev_example_export():
    import skeleton

    nml = skeleton.NML(name="My NML", scale=(11, 11, 25))
    g = nml.add_graph("A WkGraph")

    n1 = g.add_node(
        position=[0, 1, 2],
        comment="A comment",
        is_branchpoint=True,
    )
    n2 = g.add_node(
        position=[3, 1, 2],
        comment="A comment",
    )
    n3 = g.add_node(
        position=[4, 1, 2],
        comment="A comment",
    )
    g.add_edge(n1, n2)
    g.add_edge(n1, n3)

    print("connected_components", list(nx.connected_components(g.nx_graph)))

    print(g.get_node_positions())
    print(g.get_nodes())

    group = nml.add_group("Example Group")
    group.add_graph("Graph in Group").add_node(position=[10, 3, 4])
    group.add_group("Nested Group").add_graph("Graph in nested group")

    nml.write("out.nml")


def dev_example_import():
    import skeleton

    nml = skeleton.NML.from_path("in.nml")
    nml.write("exported_in.nml")
    print(nml)

    node_with_comment = nml.get_graph_by_id(1).get_node_by_id(2)
    ic(node_with_comment)
    assert node_with_comment.position == (0.0, 1.0, 2.0)
    assert node_with_comment.comment == "A comment"


def dev_example_broken():
    import skeleton

    print("wknml.open_nml", skeleton.open_nml)

    nml = skeleton.open_nml(
        "/home/philipp/coding/scalable/wk-libs/webknossos/testdata/nmls/nml_with_small_distance_nodes.nml"
    )  # or wk.open_skeleton or wk.skeleton.open
    print(nml)

    synapse_candidate_max_distance = 300  # in nm
    synapse_nodes = []

    for tree_a, tree_b in combinations(nml.flattened_graphs(), 2):
        pos_a = (
            tree_a.get_node_positions() * nml.scale
        )  # or tree_a.get_node_positions_nm?
        pos_b = tree_b.get_node_positions() * nml.scale

        print("pos_a", pos_a)
        print("pos_b", pos_b)

        index_generator = itertools.count()
        for partner_a, partner_b in pairs_within_distance(
            pos_a, pos_b, synapse_candidate_max_distance
        ):
            print("partner_a", partner_a, "partner_b", partner_b)
            synapse_nodes.append(
                skeleton.Node(
                    # todo: get rid of id.
                    # id=index_generator.__next__(),
                    position=(partner_a + partner_b) / 2,
                    # todo: allow comment
                    comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
                )
            )

    nml.add_tree("synapse candidates").add_nodes(synapse_nodes)

    # nml.write("out.nml")


def skeleton_synapse_candidate_example():
    import webknossos as wk

    nml = wk.open_nml("neurite-tracings.nml")  # or wk.open_skeleton or wk.skeleton.open

    synapse_candidate_max_distance = 300  # in nm

    synapse_nodes = []

    for tree_a, tree_b in combinations(nml.flattened_trees(), 2):
        pos_a = (
            tree_a.get_node_positions() * nml.scale
        )  # or tree_a.get_node_positions_nm?
        pos_b = tree_b.get_node_positions() * nml.scale
        # uses scipy.spatial.cKDTree.query_ball_tree internally:
        for partner_a, partner_b in wk.geometry.pairs_within_distance(
            pos_a, pos_b, synapse_candidate_max_distance
        ):
            tree.add_node(position=..., comment=...)
            synapse_nodes.append(
                wk.skeleton.Node(
                    position=(partner_a + partner_b) / 2,
                    comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
                )
            )

    nml.add_tree("synapse candidates").add_nodes(synapse_nodes)


if __name__ == "__main__":
    dev_example_import()
    # dev_example_export()

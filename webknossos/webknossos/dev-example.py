from itertools import combinations


def pairs_within_distance(pos_a, pos_b, max_distance):
    from scipy.spatial import cKDTree

    pos_a_kdtree = cKDTree(pos_a)
    pos_b_kdtree = cKDTree(pos_b)
    indexes = pos_a_kdtree.query_ball_tree(pos_b_kdtree, max_distance)
    for i in range(len(indexes)):
        for j in indexes[i]:
            yield (pos_a[i], pos_b[j])


def dev_example():
    import skeleton

    print("wknml.open_nml", skeleton.open_nml)

    nml = skeleton.open_nml(
        "/home/philipp/coding/scalable/wk-libs/webknossos/testdata/nmls/nml_with_small_distance_nodes.nml"
    )  # or wk.open_skeleton or wk.skeleton.open
    print(nml)

    synapse_candidate_max_distance = 300  # in nm
    synapse_nodes = []

    for tree_a, tree_b in combinations(nml.flattened_trees(), 2):
        pos_a = (
            tree_a.get_node_positions() * nml.scale
        )  # or tree_a.get_node_positions_nm?
        pos_b = tree_b.get_node_positions() * nml.scale

        print("pos_a", pos_a)
        print("pos_b", pos_b)

        for partner_a, partner_b in pairs_within_distance(
            pos_a, pos_b, synapse_candidate_max_distance
        ):
            print("partner_a", partner_a, "partner_b", partner_b)
            synapse_nodes.append(
                skeleton.Node(
                    position=(partner_a + partner_b) / 2,
                    comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
                )
            )

    nml.add_tree("synapse candidates").add_nodes(synapse_nodes)


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
            synapse_nodes.append(
                wk.skeleton.Node(
                    position=(partner_a + partner_b) / 2,
                    comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
                )
            )

    nml.add_tree("synapse candidates").add_nodes(synapse_nodes)


if __name__ == "__main__":
    dev_example()

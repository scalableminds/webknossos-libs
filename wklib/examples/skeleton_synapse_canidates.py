from itertools import combinations

import wklib as wk

nml = wk.open_nml("neurite-tracings.nml")  # or wk.open_skeleton or wk.skeleton.open

synapse_candidate_max_distance = 300  # in nm

synapse_nodes = []

for tree_a, tree_b in combinations(nml.flattened_trees(), 2):
    pos_a = tree_a.get_node_positions() * nml.scale  # or tree_a.get_node_positions_nm?
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

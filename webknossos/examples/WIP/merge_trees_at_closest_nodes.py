# type: ignore
# pylint: skip-file
from itertools import combinations

import networkx as nx

import webknossos as wk

nml = wk.Skeleton.load("trees-in-groups.nml")

# Probably we want to keep groups and normal trees in distinct collections (groups/trees).
# For many use-cases a common view groups_and_trees would be great, but not here:
for group in nml.groups.values():  # groups is a dict with the name as keys
    min_distance_graph = G = nx.Graph()
    for (tree_idx_a, tree_a), (tree_idx_b, tree_b) in combinations(
        enumerate(group.flattened_trees()), 2
    ):
        pos_a = (
            tree_a.get_node_positions() * nml.scale
        )  # or tree_a.get_node_positions_nm?
        pos_b = tree_b.get_node_positions() * nml.scale
        node_idx_a, node_idx_b, distance = wk.geometry.closest_pair(pos_a, pos_b)
        G.add_edge((tree_idx_a, node_idx_a), (tree_idx_b, node_idx_b), weight=distance)
    new_edges = nx.algorithms.tree.mst.minimum_spanning_edges()
    mapped_tree_ids = {}
    trees_to_merge = group.flattened_trees()
    for (tree_idx_a, node_idx_a), (tree_idx_b, node_idx_b) in new_edges:
        tree_idx_a = mapped_tree_ids.get(tree_idx_a, tree_idx_a)
        tree_idx_b = mapped_tree_ids.get(tree_idx_b, tree_idx_b)
        trees_to_merge[tree_idx_a].merge_with(
            trees_to_merge[tree_idx_b], edges=[(node_idx_a, node_idx_b)]
        )
        mapped_tree_ids[tree_idx_b] = tree_idx_a
    final_tree = trees_to_merge[mapped_tree_ids[0]]
    final_tree.name = group.name
    final_tree.group = None

    del nml.groups[group.name]
    # or
    group.delete()
    # The latter only works if everything is double-linked.
    # This also would be necessary for renaming stuff.
    # This would mean that all new objects must be added via custom functions
    # to do the double-linking. Simply dict-like insertions can't work then:
    # nml["tree-name"] = Tree()

nml.save("merged-trees.nml")

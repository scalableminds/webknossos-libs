"""
Example application: 
Finding synapse candidates with a threshold in a skeleton
annotation where each neuron is represented/reconstructed as one long tree 
of many nodes placed reguarly along its axon/dendrite paths.

Method:
Load an NML file and consider all pairs of trees.
For each tree pair, find the node pairs that have a distance
lower than a given threshold.
For these candidates, new annotations are created which contain a node at the
center position between the input nodes.
"""

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np
from scipy.spatial import cKDTree

import webknossos as wk


def pairs_within_distance(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    max_distance: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    pos_a_kdtree = cKDTree(pos_a)
    pos_b_kdtree = cKDTree(pos_b)
    indexes = pos_a_kdtree.query_ball_tree(pos_b_kdtree, max_distance)
    for i in range(len(indexes)):
        for j in indexes[i]:
            yield (pos_a[i], pos_b[j])


def main() -> None:
    skeleton = wk.Skeleton.load("testdata/nmls/nml_with_small_distance_nodes.nml")

    synapse_candidate_max_distance = 150  # in nm

    input_trees = list(skeleton.flattened_trees())
    synapse_parent_group = skeleton.add_group("all synapse candidates")

    for tree_a, tree_b in combinations(input_trees, 2):
        positions_a = tree_a.get_node_positions() * skeleton.voxel_size
        positions_b = tree_b.get_node_positions() * skeleton.voxel_size

        synapse_tree = synapse_parent_group.add_tree(
            f"synapse candidates ({tree_a.name}-{tree_b.name})"
        )

        for partner_a, partner_b in pairs_within_distance(
            positions_a, positions_b, synapse_candidate_max_distance
        ):
            synapse_tree.add_node(
                position=np.round((partner_a + partner_b) / skeleton.voxel_size / 2),
                comment=f"{tree_a.name} ({tree_a.id}) <-> {tree_b.name} ({tree_b.id})",
            )

    # skeleton can be used for further processing or written to a file:
    # skeleton.save("output_path.nml")


if __name__ == "__main__":
    main()

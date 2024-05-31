from typing import Tuple

import numpy as np

import webknossos as wk


def calculate_path_length(annotation_url: str, auth_token: str) -> None:
    with wk.webknossos_context(token=auth_token):
        # Download a annotation directly from the WEBKNOSSOS server
        annotation = wk.Annotation.download(
            annotation_url,
        )

        skeleton = annotation.skeleton
        voxel_size = annotation.voxel_size

    # Iterate over all the tree in a skeleton and calculate their path length
    for tree in skeleton.flattened_trees():
        path_length = calculate_path_length_for_tree(tree, voxel_size)

        # Log the results :-)
        print(f"Tree {tree.name} has a path length of {path_length:.2f} nm")


def calculate_path_length_for_tree(
    tree: wk.Tree, voxel_size: Tuple[float, float, float]
) -> float:
    # Auxiliary method calculate the maximum path length of a given tree
    # Assumes that the annotation does not contain any cycles

    assert (
        len(tree.nodes) > 1
    ), "Each tree should have at least two nodes to calculate the path length"
    result = 0

    # Iterate over all edges
    for source_node, target_node in tree.edges:
        diff_vector = np.array(source_node.position) - np.array(target_node.position)
        scaled_diff_vector = diff_vector * voxel_size
        edge_length = np.sqrt(scaled_diff_vector.dot(scaled_diff_vector))
        result += edge_length

    return result


if __name__ == "__main__":
    # Authentication and API token for your account
    # Get it at https://webknossos.org/auth/token
    TOKEN = "YOUR-token"

    # A WEBKNOSOS URL containing the skeleton annotation
    ANNOTATION_URL = "https://webknossos.org/annotations/12345678"

    calculate_path_length(ANNOTATION_URL, TOKEN)

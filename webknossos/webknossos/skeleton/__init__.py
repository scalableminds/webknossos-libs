import numpy as np
from typing import List, Tuple, Optional

import skeleton.legacy as legacy_wknml

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]


class Node:
    def __init__(self, position, comment=None):

        self.position = position
        self.comment = comment


class Edge:
    def __init__(self):

        pass


class Skeleton:
    """
    Contains a collection of nodes and edges.
    """

    def __init__(self, legacy_tree):

        self._legacy_tree = legacy_tree

    def get_node_positions(self):
        return np.array([node.position for node in self._legacy_tree.nodes])

    def add_nodes(self, nodes):

        # todo: don't loop to concat
        for node in nodes:
            self._legacy_tree.nodes.append(node)

    @property
    def id(self) -> int:
        return self._legacy_tree.id

    @property
    def color(self) -> Vector4:
        return self._legacy_tree.color

    @property
    def name(self) -> str:
        return self._legacy_tree.name

    @property
    def nodes(self) -> List[Node]:
        return [Node(legacy_node) for legacy_node in self._legacy_tree.nodes]

    @property
    def edges(self) -> List[Edge]:
        return [Node(legacy_edge) for legacy_edge in self._legacy_tree.edges]

    @property
    def groupId(self) -> Optional[int]:
        return self._legacy_tree.groupId


class NML:
    """
    Contains groups and skeletons.
    """

    def __init__(self, legacy_nml):

        self._legacy_nml = legacy_nml

        self._flat_skeletons = [Skeleton(tree) for tree in self._legacy_nml.trees]

    def flattened_trees(self):
        return self._flat_skeletons

    def add_tree(self, name):

        new_skeleton = Skeleton(
            legacy_wknml.Tree(
                id=self.get_max_skeleton_id() + 1,
                color=(255, 0, 0, 1),
                name=name,
                nodes=[],
                edges=[],
                groupId=None,
            )
        )
        self._flat_skeletons.append(new_skeleton)

        return new_skeleton

    def get_max_skeleton_id(self):

        return max(tree.id for tree in self.flattened_trees())

    @staticmethod
    def from_path(file_path):

        return NML(legacy_wknml.parse_nml(file_path))

    @property
    def scale(self):

        return self._legacy_nml.parameters.scale

    def write(self):
        with open("out.nml", "wb") as f:
            legacy_wknml.write_nml(f, nml)


def open_nml(file_path):
    return NML.from_path(file_path)

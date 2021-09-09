import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy
from loxun import XmlWriter

import attr

# @attr.define
# class LayerProperties:
#     name: str
#     category: str
#     bounding_box: BoundingBox
#     element_class: str
#     wkw_resolutions: List[MagViewProperties]
#     data_format: str
#     num_channels: Optional[int] = None
#     default_view_configuration: Optional[LayerViewConfiguration] = None

import skeleton.legacy as legacy_wknml

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]


class Node:
    def __init__(self, position, comment=None):

        self.position = position
        self.comment = comment

        self.radius: Optional[float] = None
        self.rotation: Optional[Vector3] = None
        self.inVp: Optional[int] = None
        self.inMag: Optional[int] = None
        self.bitDepth: Optional[int] = None
        self.interpolation: Optional[bool] = None
        self.time: Optional[int] = None

    def as_legacy_node(self):

        # todo: id and comment
        return legacy_wknml.Node(id=-1, position=self.position)

    @staticmethod
    def from_legacy(legacy_node):

        return Node(position=legacy_node.position)

    @property
    def id(self):
        return -1

    def _dump(self, xf: XmlWriter):
        # Adapted from __dump_node

        node = self
        print("node.position", node.position)
        attributes = {
            "id": str(node.id),
            "x": str(node.position[0]),
            "y": str(node.position[1]),
            "z": str(node.position[2]),
        }

        if node.radius is not None:
            attributes["radius"] = str(node.radius)

        if node.rotation is not None:
            attributes["rotX"] = str(node.rotation[0])
            attributes["rotY"] = str(node.rotation[1])
            attributes["rotZ"] = str(node.rotation[2])

        if node.inVp is not None:
            attributes["inVp"] = str(node.inVp)

        if node.inMag is not None:
            attributes["inMag"] = str(node.inMag)

        if node.bitDepth is not None:
            attributes["bitDepth"] = str(node.bitDepth)

        if node.interpolation is not None:
            attributes["interpolation"] = str(node.interpolation)

        if node.time is not None:
            attributes["time"] = str(node.time)

        xf.tag("node", attributes)


class Edge:
    def __init__(self, source: int, target: int):

        self.source = source
        self.target = target

    def _dump(self, xf: XmlWriter):
        xf.tag("edge", {"source": str(self.source), "target": str(self.target)})

    @staticmethod
    def from_legacy(legacy_edge):

        return Edge(source=legacy_edge.source, target=legacy_edge.target)


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

    def get_max_node_id(self):

        return max(node.id for node in self._legacy_tree.nodes)

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
        return [
            Node.from_legacy(legacy_node) for legacy_node in self._legacy_tree.nodes
        ]

    @property
    def edges(self) -> List[Edge]:
        return [
            Edge.from_legacy(legacy_edge) for legacy_edge in self._legacy_tree.edges
        ]

    @property
    def groupId(self) -> Optional[int]:
        return self._legacy_tree.groupId

    def _dump(self, xf: XmlWriter):
        tree = self
        attributes = {
            "id": str(tree.id),
            "color.r": str(tree.color[0]),
            "color.g": str(tree.color[1]),
            "color.b": str(tree.color[2]),
            "color.a": str(tree.color[3]),
            "name": tree.name,
        }

        if tree.groupId is not None:
            attributes["groupId"] = str(tree.groupId)

        xf.startTag("thing", attributes)
        xf.startTag("nodes")
        for n in tree.nodes:
            n._dump(xf)
        xf.endTag()  # nodes
        xf.startTag("edges")
        for e in tree.edges:
            e._dump(xf)
        xf.endTag()  # edges
        xf.endTag()  # thing


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

        self._legacy_nml = deepcopy(self._legacy_nml)._replace(
            trees=self._flat_skeletons
        )

        return new_skeleton

    def get_max_skeleton_id(self):

        return max(tree.id for tree in self.flattened_trees())

    def get_max_node_id(self):

        return max(tree.get_max_node_id() for tree in self.flattened_trees())

    @staticmethod
    def from_path(file_path):

        return NML(legacy_wknml.parse_nml(file_path))

    @property
    def scale(self):

        return self._legacy_nml.parameters.scale

    def ensure_valid_ids(self):

        # max_node_id = self.get_max_node_id()
        pass

    def write(self, out_path):
        with open(out_path, "wb") as f:
            legacy_wknml.write_nml(f, self._legacy_nml)


def open_nml(file_path):
    return NML.from_path(file_path)

import numpy as np
from typing import List, Tuple, Optional, Union
import itertools
import networkx as nx


import attr


import skeleton.legacy as legacy_wknml
import skeleton.legacy.nml_generation as nml_generation

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union["Group", "WkGraph"]


@attr.define(eq=False)
class Group:
    name: str
    children: List[GroupOrGraph]


@attr.define(eq=False)
class Node:
    position: Vector3
    id: int = None
    comment: Optional = None
    radius: Optional[float] = None
    rotation: Optional[Vector3] = None
    inVp: Optional[int] = None
    inMag: Optional[int] = None
    bitDepth: Optional[int] = None
    interpolation: Optional[bool] = None
    time: Optional[int] = None

    is_branchpoint: bool = False


@attr.define(eq=False)
class WkGraph:
    """
    Contains a collection of nodes and edges.
    """

    id: int
    name: str
    color: Optional[Vector4] = None
    nx_graph: nx.Graph = nx.Graph()

    def get_nodes(self):

        return self.nx_graph.nodes

    def get_node_positions(self):
        return np.array([node.position for node in self.nx_graph.nodes])

    def add_node(self, node):

        if node.id is None:
            node.id = self.get_max_node_id() + 1

        self.nx_graph.add_node(node)

    def add_edge(self, node_1, node_2):

        self.nx_graph.add_edge(node_1, node_2)

    def add_nodes(self, nodes):

        self.nx_graph.add_nodes_from(nodes)

    def get_max_node_id(self):

        # Chain with [0] since max is not defined on empty sequences
        return max(itertools.chain((node.id for node in self.nx_graph.nodes), [0]))


def iterate_graphs(wk_graphs_or_groups: List[GroupOrGraph]):
    for wk_graph_or_group in wk_graphs_or_groups:
        if isinstance(wk_graph_or_group, WkGraph):
            yield wk_graph_or_group
        else:
            yield from iterate_graphs(wk_graph_or_group.children)


def get_graphs_as_dict(wk_graphs_or_groups: List[GroupOrGraph], dictionary):

    for wk_graph_or_group in wk_graphs_or_groups:
        if isinstance(wk_graph_or_group, WkGraph):
            wk_graph = wk_graph_or_group
            dictionary[wk_graph.name] = wk_graph.nx_graph
        else:
            wk_group = wk_graph_or_group
            inner_dictionary = {}
            get_graphs_as_dict(wk_group.children, inner_dictionary)
            dictionary[wk_group.name] = inner_dictionary


@attr.define(eq=False)
class NML:
    """
    Contains groups and skeletons.
    """

    name: str
    scale: Vector3
    offset: Optional[Vector3] = None
    time: Optional[int] = None
    editPosition: Optional[Vector3] = None
    editRotation: Optional[Vector3] = None
    zoomLevel: Optional[float] = None
    taskBoundingBox: Optional[IntVector6] = None
    userBoundingBoxes: Optional[List[IntVector6]] = None

    _wk_graphs_or_groups: List[GroupOrGraph] = []

    def flattened_graphs(self):

        return iterate_graphs(self._wk_graphs_or_groups)

    def add_graph(self, name):

        new_graph = WkGraph(
            id=self.get_max_graph_id() + 1,
            name=name,
        )
        self._wk_graphs_or_groups.append(new_graph)

        return new_graph

    def get_max_graph_id(self):

        # Chain with [0] since max is not defined on empty sequences
        return max(itertools.chain((tree.id for tree in self.flattened_graphs()), [0]))

    def get_max_node_id(self):

        # Chain with [0] since max is not defined on empty sequences
        return max(
            itertools.chain(
                (tree.get_max_node_id() for tree in self.flattened_graphs()),
                [0],
            )
        )

    # @staticmethod
    # def from_path(file_path):

    #     return NML(legacy_wknml.parse_nml(file_path))

    # @property
    # def scale(self):

    #     return self._legacy_nml.parameters.scale

    # def ensure_valid_ids(self):

    #     # max_node_id = self.get_max_node_id()
    #     pass

    def write(self, out_path):

        graphs_as_dict = self.get_graphs_as_dict()
        print("graphs_as_dict", graphs_as_dict)
        legacy_nml = nml_generation.generate_nml(
            [g for g in self.flattened_graphs()],
            self._get_legacy_parameters(),
            globalize_ids=False,
        )

        with open(out_path, "wb") as f:
            legacy_wknml.write_nml(f, legacy_nml)

    def get_graphs_as_dict(self):

        dictionary = {}
        get_graphs_as_dict(self._wk_graphs_or_groups, dictionary)
        return dictionary

    def _get_legacy_parameters(self):

        return {
            "name": self.name,
            "scale": self.scale,
            "offset": self.offset,
            "time": self.time,
            "editPosition": self.editPosition,
            "editRotation": self.editRotation,
            "zoomLevel": self.zoomLevel,
            "taskBoundingBox": self.taskBoundingBox,
            "userBoundingBoxes": self.userBoundingBoxes,
        }


def open_nml(file_path):
    return NML.from_path(file_path)

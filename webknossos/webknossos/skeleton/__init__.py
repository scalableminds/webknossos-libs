import numpy as np
from typing import List, Tuple, Optional, Union, Generator
import itertools
import networkx as nx


import attr


import skeleton.legacy as legacy_wknml
import skeleton.legacy.nml_generation as nml_generation

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union["Group", "WkGraph"]

nml_id_generator = itertools.count()


@attr.define()
class Group:
    id: int = attr.ib(init=False)
    name: str
    children: List[GroupOrGraph]
    _root: "NML"
    # _parent: Union["Group", "NML"]

    def __attrs_post_init__(self):
        self.id = self._root.nml_element_id_generator.__next__()

    def add_graph(self, name: str):

        new_graph = WkGraph(name=name, root=self._root)
        self.children.append(new_graph)

        return new_graph

    def add_group(self, name: str, children: Optional[List[GroupOrGraph]] = None):

        new_group = Group(name, children or [], root=self._root)
        self.children.append(new_group)
        return new_group

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

    def flattened_graphs(self):
        for child in self.children:
            if isinstance(child, Group):
                yield from child.flattened_graphs()
            else:
                yield child

    def __hash__(self):
        return self.id


@attr.define()
class Node:
    position: Vector3
    _root: "NML"
    id: int = attr.ib(init=False)
    comment: Optional = None
    radius: Optional[float] = None
    rotation: Optional[Vector3] = None
    inVp: Optional[int] = None
    inMag: Optional[int] = None
    bitDepth: Optional[int] = None
    interpolation: Optional[bool] = None
    time: Optional[int] = None

    is_branchpoint: bool = False
    _enforce_given_id: Optional[int] = None

    def __attrs_post_init__(self):
        if self._enforce_given_id is not None:
            self.id = self._enforce_given_id
        else:
            self.id = self._root.nml_element_id_generator.__next__()

    def __hash__(self):
        return self.id


@attr.define()
class WkGraph:
    """
    Contains a collection of nodes and edges.
    """

    name: str
    _root: "NML"
    color: Optional[Vector4] = None
    id: int = attr.ib(init=False)
    nx_graph: nx.Graph = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.nx_graph = nx.Graph()
        self.id = self._root.nml_element_id_generator.__next__()

    def get_nodes(self):

        return self.nx_graph.nodes

    def get_node_positions(self):
        return np.array([node.position for node in self.nx_graph.nodes])

    def add_node(self, *args, **kwargs):

        node = Node(*args, **kwargs, root=self._root)
        self.nx_graph.add_node(node)
        return node

    def add_edge(self, node_1, node_2):

        self.nx_graph.add_edge(node_1, node_2)

    def get_max_node_id(self):

        # Chain with [0] since max is not defined on empty sequences
        return max(itertools.chain((node.id for node in self.nx_graph.nodes), [0]))

    def __hash__(self):
        return self.id


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


@attr.define()
class NML:
    """
    Contains groups and skeletons.
    """

    name: str
    id: int = attr.ib(init=False)
    scale: Vector3
    offset: Optional[Vector3] = None
    time: Optional[int] = None
    editPosition: Optional[Vector3] = None
    editRotation: Optional[Vector3] = None
    zoomLevel: Optional[float] = None
    taskBoundingBox: Optional[IntVector6] = None
    userBoundingBoxes: Optional[List[IntVector6]] = None

    root_group: Group = attr.ib(init=False)
    nml_element_id_generator: Generator[None, int, None] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.id = nml_id_generator.__next__()
        self.nml_element_id_generator = itertools.count()
        self.root_group = Group("Root", [], self)

    def flattened_graphs(self):

        return self.root_group.flattened_graphs()

    def add_graph(self, name: str):

        return self.root_group.add_graph(name)

    def add_group(self, name: str, children: Optional[List[GroupOrGraph]] = None):

        return self.root_group.add_group(name, children)

    def get_max_graph_id(self):

        return self.root_group.get_max_graph_id()

    def get_max_node_id(self):

        return self.root_group.get_max_node_id()

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

        # graphs_as_dict = self.get_graphs_as_dict()
        # print("graphs_as_dict", graphs_as_dict)

        print(
            "[g for g in self.flattened_graphs()]", [g for g in self.flattened_graphs()]
        )

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

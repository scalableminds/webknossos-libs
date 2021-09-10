import numpy as np
from typing import List, Tuple, Optional, Union, Generator
import itertools
import networkx as nx
from icecream import ic


import attr


import webknossos.skeleton.legacy as legacy_wknml
import webknossos.skeleton.legacy.nml_generation as nml_generation

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
    _nml: "NML"
    is_root_group: bool = False
    # _parent: Union["Group", "NML"]
    _enforce_id: Optional[int] = None

    def __attrs_post_init__(self):

        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.nml_element_id_generator.__next__()

    def add_graph(self, name: str, enforce_id: Optional[int] = None, **kwargs):

        new_graph = WkGraph(
            name=name, nml=self._nml, group_id=self.id, enforce_id=enforce_id, **kwargs
        )
        self.children.append(new_graph)

        return new_graph

    def add_group(
        self, name: str, children: Optional[List[GroupOrGraph]] = None, enforce_id=None
    ):

        new_group = Group(name, children or [], nml=self._nml, enforce_id=enforce_id)
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

    def flattened_groups(self):
        for child in self.children:
            if isinstance(child, Group):
                yield child
                yield from child.flattened_groups()

    def as_nml_group(self):

        return legacy_wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self.children if isinstance(g, Group)],
        )

    def __hash__(self):
        return hash((self._nml.id, self.id))


@attr.define()
class Node:
    position: Vector3
    _nml: "NML"
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
    branchpoint_time: Optional[int] = None
    _enforce_id: Optional[int] = None

    def __attrs_post_init__(self):
        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.nml_element_id_generator.__next__()

    def __hash__(self):
        return hash((self._nml.id, self.id))

    def __eq__(self, other):
        return hash(self) == hash(other)


@attr.define()
class DummyNode:
    id: int
    _nml: "NML"

    def __hash__(self):
        return hash((self._nml.id, self.id))

    def __eq__(self, other):
        return hash(self) == hash(other)


@attr.define()
class WkGraph:
    """
    Contains a collection of nodes and edges.
    """

    name: str
    _nml: "NML"
    color: Optional[Vector4] = None
    id: int = attr.ib(init=False)
    nx_graph: nx.Graph = attr.ib(init=False)
    group_id: Optional[int] = None

    _enforce_id: Optional[int] = None

    def __attrs_post_init__(self):
        self.nx_graph = nx.Graph()
        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.nml_element_id_generator.__next__()

    def get_nodes(self):

        return self.nx_graph.nodes

    def get_node_positions(self):
        return np.array([node.position for node in self.nx_graph.nodes])

    def get_node_by_id(self, node_id: int) -> Node:

        # Todo: Use hashed access
        for node in self.get_nodes():
            if node.id == node_id:
                return node
        assert ValueError(f"No node with id {node_id} was found")

    def add_node(self, *args, **kwargs):

        node = Node(*args, **kwargs, nml=self._nml)
        self.nx_graph.add_node(node)
        return node

    def add_edge(self, node_1, node_2):

        self.nx_graph.add_edge(node_1, node_2)

    def get_max_node_id(self):

        # Chain with [0] since max is not defined on empty sequences
        return max(itertools.chain((node.id for node in self.nx_graph.nodes), [0]))

    def __hash__(self):
        return hash((self._nml.id, self.id))


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
        self.root_group = Group("Root", [], self, is_root_group=False)

    def flattened_graphs(self):

        return self.root_group.flattened_graphs()

    def get_graph_by_id(self, graph_id: int) -> WkGraph:

        # Todo: Use hashed access
        for graph in self.root_group.flattened_graphs():
            if graph.id == graph_id:
                return graph
        assert ValueError(f"No graph with id {graph_id} was found")

    def add_graph(self, name: str, **kwargs):

        return self.root_group.add_graph(name, **kwargs)

    def add_group(self, name: str, children: Optional[List[GroupOrGraph]] = None):

        return self.root_group.add_group(name, children)

    def flattened_groups(self):

        return self.root_group.flattened_groups()

    def get_max_graph_id(self):

        return self.root_group.get_max_graph_id()

    def get_max_node_id(self):

        return self.root_group.get_max_node_id()

    @staticmethod
    def from_path(file_path):

        return NML.from_legacy_nml(legacy_wknml.parse_nml(file_path))

    @staticmethod
    def from_legacy_nml(legacy_nml):
        nml = NML(legacy_nml.parameters.name, legacy_nml.parameters.scale)

        groups_by_id = {}

        def visit_groups(legacy_groups, current_group):

            for legacy_group in legacy_groups:
                sub_group = current_group.add_group(
                    name=legacy_group.name, enforce_id=legacy_group.id
                )
                groups_by_id[sub_group.id] = sub_group
                visit_groups(legacy_group.children, sub_group)

        visit_groups(legacy_nml.groups, nml.root_group)
        for tree in legacy_nml.trees:
            if tree.groupId is None:
                new_graph = nml.root_group.add_graph(tree.name)
            else:
                new_graph = groups_by_id[tree.groupId].add_graph(tree.name)
            NML.nml_tree_to_graph(legacy_nml, nml, new_graph, tree)

        nml.write("out_only_groups.nml")

        # nml_parameters = nml.parameters
        # parameter_dict = {}

        # parameter_list = [
        #     "name",
        #     "scale",
        #     "offset",
        #     "time",
        #     "editPosition",
        #     "editRotation",
        #     "zoomLevel",
        #     "taskBoundingBox",
        #     "userBoundingBoxes",
        # ]

        # for parameter in parameter_list:
        #     if getattr(nml_parameters, parameter) is not None:
        #         parameter_dict[parameter] = getattr(nml_parameters, parameter)

        # for comment in nml.comments:
        #     for tree in group:
        #         if comment.node in tree.nodes:
        #             tree.nodes[comment.node]["comment"] = comment.content

        # for branchpoint in nml.branchpoints:
        #     for group in group_dict.values():
        #         for tree in group:
        #             if branchpoint.id in tree.nodes:
        #                 tree.nodes[branchpoint.id]["branchpoint"] = branchpoint.time

        # return group_dict, parameter_dict
        print(nml)
        return nml

    def nml_tree_to_graph(legacy_nml, new_nml, new_graph, legacy_tree) -> nx.Graph:
        """
        A utility to convert a single wK Tree object into a [NetworkX graph object](https://networkx.org/).
        """

        optional_attribute_list = [
            "rotation",
            "inVp",
            "inMag",
            "bitDepth",
            "interpolation",
            "time",
        ]

        new_graph.color = legacy_tree.color
        new_graph.name = legacy_tree.name
        new_graph.group_id = legacy_tree.groupId

        node_by_id = {}

        for legacy_node in legacy_tree.nodes:
            node_id = legacy_node.id
            node_by_id[node_id] = new_graph.add_node(
                position=legacy_node.position,
                enforce_id=node_id,
                radius=legacy_node.radius,
            )

            for optional_attribute in optional_attribute_list:
                if getattr(legacy_node, optional_attribute) is not None:
                    setattr(
                        node_by_id[node_id],
                        optional_attribute,
                        getattr(legacy_node, optional_attribute),
                    )

        for edge in legacy_tree.edges:
            source_node = node_by_id[edge.source]
            target_node = node_by_id[edge.target]

            new_graph.add_edge(source_node, target_node)

        for comment in legacy_nml.comments:
            # Unfortunately, legacy_nml.comments is not grouped by tree which is
            # why we currently go over all comments in the NML and only attach
            # the ones for the current tree.
            # Todo: Implement a more efficient way.
            if comment.node in node_by_id:
                node_by_id[comment.node].comment = comment.content

        for branchpoint in legacy_nml.branchpoints:
            # Unfortunately, legacy_nml.branchpoints is not grouped by tree which is
            # why we currently go over all branchpoints in the NML and only attach
            # the ones for the current tree.
            # Todo: Implement a more efficient way.
            if branchpoint.id in node_by_id:
                node = node_by_id[branchpoint.id]
                node.is_branchpoint = True
                if branchpoint.time != 0:
                    node.branchpoint_time = branchpoint.time

        return new_graph

    def write(self, out_path):

        legacy_nml = nml_generation.generate_nml(
            self.root_group,
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

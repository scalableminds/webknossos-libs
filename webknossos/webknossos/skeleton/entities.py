import colorsys
import itertools
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import attr
import networkx as nx
import numpy as np
from icecream import ic

import webknossos.skeleton.legacy as legacy_wknml
from webknossos.skeleton.exporter import NMLExporter
from webknossos.skeleton.legacy import NML as LegacyNML
from webknossos.skeleton.legacy import Group as LegacyGroup
from webknossos.skeleton.legacy import Tree as LegacyTree

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

Number = Union[int, float]

GroupOrGraph = Union["Group", "WkGraph"]

nml_id_generator = itertools.count()


def opt_vector3_as_float(
    vec: Optional[Tuple[Number, Number, Number]]
) -> Optional[Vector3]:
    if vec is None:
        return None
    return (
        float(vec[0]),
        float(vec[1]),
        float(vec[2]),
    )


def vector3_as_float(vec: Tuple[Number, Number, Number]) -> Vector3:
    return (
        float(vec[0]),
        float(vec[1]),
        float(vec[2]),
    )


@attr.define()
class Group:
    id: int = attr.ib(init=False)
    name: str
    children: List[GroupOrGraph]
    _nml: "NML"
    is_root_group: bool = False
    _enforce_id: Optional[int] = None

    def __attrs_post_init__(self) -> None:

        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.element_id_generator.__next__()

    def add_graph(
        self,
        name: str,
        color: Optional[Vector4] = None,
        _nml: Optional["NML"] = None,
        _enforce_id: Optional[int] = None,
    ) -> "WkGraph":

        new_graph = WkGraph(
            name=name,
            color=color,
            group_id=self.id,
            nml=_nml or self._nml,
            enforce_id=_enforce_id,
        )
        self.children.append(new_graph)

        return new_graph

    def add_group(
        self,
        name: str,
        children: Optional[List[GroupOrGraph]] = None,
        _enforce_id: int = None,
    ) -> "Group":

        new_group = Group(name, children or [], nml=self._nml, enforce_id=_enforce_id)  # type: ignore
        self.children.append(new_group)
        return new_group

    def get_total_node_count(self) -> int:
        return sum(
            len(synapse_graph.get_nodes()) for synapse_graph in self.flattened_graphs()
        )

    def get_max_graph_id(self) -> int:
        # Chain with [0] since max is not defined on empty sequences
        return max(
            itertools.chain((graph.id for graph in self.flattened_graphs()), [0])
        )

    def get_max_node_id(self) -> int:
        # Chain with [0] since max is not defined on empty sequences
        return max(
            itertools.chain(
                (graph.get_max_node_id() for graph in self.flattened_graphs()),
                [0],
            )
        )

    def flattened_graphs(self) -> Generator["WkGraph", None, None]:
        for child in self.children:
            if isinstance(child, Group):
                yield from child.flattened_graphs()
            else:
                yield child

    def flattened_groups(self) -> Generator["Group", None, None]:
        for child in self.children:
            if isinstance(child, Group):
                yield child
                yield from child.flattened_groups()

    def as_legacy_group(self) -> "LegacyGroup":  # type: ignore

        return legacy_wknml.Group(
            self.id,
            self.name,
            children=[
                g.as_legacy_group() for g in self.children if isinstance(g, Group)
            ],
        )

    def __hash__(self) -> int:
        return hash((self._nml.id, self.id))


@attr.define()
class Node:
    position: Vector3
    _nml: "NML"
    id: int = attr.ib(init=False)
    comment: Optional[str] = None
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

    def __attrs_post_init__(self) -> None:
        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.element_id_generator.__next__()

        self.position = vector3_as_float(self.position)

    def __hash__(self) -> int:
        return hash((self._nml.id, self.id))

    def __eq__(self, other: object) -> bool:
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

    def __attrs_post_init__(self) -> None:
        self.nx_graph = nx.Graph()
        if self._enforce_id is not None:
            self.id = self._enforce_id
        else:
            self.id = self._nml.element_id_generator.__next__()

    # todo: how to type this?
    def get_nodes(self) -> Any:

        return self.nx_graph.nodes

    def get_node_positions(self) -> np.ndarray:
        return np.array([node.position for node in self.nx_graph.nodes])

    def get_node_by_id(self, node_id: int) -> Node:

        # Todo: Use hashed access
        for node in self.get_nodes():
            if node.id == node_id:
                return node
        raise ValueError(f"No node with id {node_id} was found")

    def add_node(
        self,
        position: Vector3,
        comment: Optional[str] = None,
        radius: Optional[float] = None,
        rotation: Optional[Vector3] = None,
        inVp: Optional[int] = None,
        inMag: Optional[int] = None,
        bitDepth: Optional[int] = None,
        interpolation: Optional[bool] = None,
        time: Optional[int] = None,
        is_branchpoint: bool = False,
        branchpoint_time: Optional[int] = None,
        _enforce_id: Optional[int] = None,
        _nml: Optional["NML"] = None,
    ) -> Node:
        node = Node(
            position=position,
            comment=comment,
            radius=radius,
            rotation=rotation,
            inVp=inVp,
            inMag=inMag,
            bitDepth=bitDepth,
            interpolation=interpolation,
            time=time,
            is_branchpoint=is_branchpoint,
            branchpoint_time=branchpoint_time,
            enforce_id=_enforce_id,
            nml=_nml or self._nml,
        )
        self.nx_graph.add_node(node)
        return node

    def add_edge(self, node_1: Node, node_2: Node) -> None:

        self.nx_graph.add_edge(node_1, node_2)

    def get_max_node_id(self) -> int:

        # Chain with [0] since max is not defined on empty sequences
        return max(itertools.chain((node.id for node in self.nx_graph.nodes), [0]))

    def __hash__(self) -> int:
        return hash((self._nml.id, self.id))


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
    edit_position: Optional[Vector3] = None
    edit_rotation: Optional[Vector3] = None
    zoom_level: Optional[float] = None
    task_bounding_box: Optional[IntVector6] = None
    user_bounding_boxes: Optional[List[IntVector6]] = None

    root_group: Group = attr.ib(init=False)
    element_id_generator: Iterator[int] = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.id = nml_id_generator.__next__()
        self.element_id_generator = itertools.count()
        self.root_group = Group(name="Root", children=[], nml=self, is_root_group=False)
        self.scale = vector3_as_float(self.scale)
        self.time = int(str(self.time))  # typing: ignore
        self.offset = opt_vector3_as_float(self.offset)
        self.edit_position = opt_vector3_as_float(self.edit_position)
        self.edit_rotation = opt_vector3_as_float(self.edit_rotation)

    def flattened_graphs(self) -> Generator["WkGraph", None, None]:

        return self.root_group.flattened_graphs()

    def get_graph_by_id(self, graph_id: int) -> WkGraph:

        # Todo: Use hashed access
        for graph in self.root_group.flattened_graphs():
            if graph.id == graph_id:
                return graph
        raise ValueError(f"No graph with id {graph_id} was found")

    def add_graph(
        self,
        name: str,
        color: Optional[Vector4] = None,
        _nml: Optional["NML"] = None,
        _enforce_id: Optional[int] = None,
    ) -> "WkGraph":

        return self.root_group.add_graph(
            name,
            color,
            _nml,
            _enforce_id,
        )

    def add_group(
        self, name: str, children: Optional[List[GroupOrGraph]] = None
    ) -> "Group":

        return self.root_group.add_group(name, children)

    def get_total_node_count(self) -> int:

        return self.root_group.get_total_node_count()

    def flattened_groups(self) -> Generator["Group", None, None]:

        return self.root_group.flattened_groups()

    def get_max_graph_id(self) -> int:

        return self.root_group.get_max_graph_id()

    def get_max_node_id(self) -> int:

        return self.root_group.get_max_node_id()

    @staticmethod
    def from_path(file_path: str) -> "NML":

        with open(file_path, "rb") as file_handle:
            return NML.from_legacy_nml(legacy_wknml.parse_nml(file_handle))

    @staticmethod
    def from_legacy_nml(legacy_nml: LegacyNML) -> "NML":
        nml = NML(
            name=legacy_nml.parameters.name,
            scale=legacy_nml.parameters.scale,
            offset=legacy_nml.parameters.offset,
            time=legacy_nml.parameters.time,
            edit_position=legacy_nml.parameters.editPosition,
            edit_rotation=legacy_nml.parameters.editRotation,
            zoom_level=legacy_nml.parameters.zoomLevel,
            task_bounding_box=legacy_nml.parameters.taskBoundingBox,
            user_bounding_boxes=legacy_nml.parameters.userBoundingBoxes,
        )

        groups_by_id = {}

        def visit_groups(
            legacy_groups: List[LegacyGroup], current_group: Group
        ) -> None:

            for legacy_group in legacy_groups:
                sub_group = current_group.add_group(
                    name=legacy_group.name, _enforce_id=legacy_group.id
                )
                groups_by_id[sub_group.id] = sub_group
                visit_groups(legacy_group.children, sub_group)

        visit_groups(legacy_nml.groups, nml.root_group)
        for legacy_tree in legacy_nml.trees:
            if legacy_tree.groupId is None:
                new_graph = nml.root_group.add_graph(
                    legacy_tree.name, _enforce_id=legacy_tree.id
                )
            else:
                new_graph = groups_by_id[legacy_tree.groupId].add_graph(
                    legacy_tree.name, _enforce_id=legacy_tree.id
                )
            NML.nml_tree_to_graph(legacy_nml, new_graph, legacy_tree)

        nml.write("out_only_groups.nml")

        max_id = max(nml.get_max_graph_id(), nml.get_max_node_id())
        nml.element_id_generator = itertools.count(max_id + 1)

        return nml

    @staticmethod
    def nml_tree_to_graph(
        legacy_nml: LegacyNML,
        new_graph: "WkGraph",
        legacy_tree: LegacyTree,
    ) -> nx.Graph:
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
                _enforce_id=node_id,
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

    def write(self, out_path: str) -> None:

        legacy_nml = NMLExporter.generate_nml(
            self.root_group,
            self._get_legacy_parameters(),
        )

        with open(out_path, "wb") as f:
            legacy_wknml.write_nml(f, legacy_nml)

    def _get_legacy_parameters(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "scale": self.scale,
            "offset": self.offset,
            "time": self.time,
            "editPosition": self.edit_position,
            "editRotation": self.edit_rotation,
            "zoomLevel": self.zoom_level,
            "taskBoundingBox": self.task_bounding_box,
            "userBoundingBoxes": self.user_bounding_boxes,
        }

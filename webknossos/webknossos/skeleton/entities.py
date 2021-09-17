import itertools
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import attr
import networkx as nx
import numpy as np

import webknossos.skeleton.nml as wknml
from webknossos.skeleton.exporter import NMLExporter
from webknossos.skeleton.nml import NML as WkNML
from webknossos.skeleton.nml import Group as NmlGroup
from webknossos.skeleton.nml import Tree as NmlTree

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union["Group", "Graph"]

nml_id_generator = itertools.count()


@attr.define()
class Group:
    _id: int = attr.ib(init=False)
    name: str
    children: List[GroupOrGraph]
    _nml: "Skeleton"
    is_root_group: bool = False
    _enforced_id: Optional[int] = None

    def __attrs_post_init__(self) -> None:

        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._nml.element_id_generator.__next__()

    @property
    def id(self):
        return self._id

    def add_graph(
        self,
        name: str,
        color: Optional[Vector4] = None,
        _nml: Optional["Skeleton"] = None,
        _enforced_id: Optional[int] = None,
    ) -> "Graph":

        new_graph = Graph(
            name=name,
            color=color,
            group_id=self.id,
            nml=_nml or self._nml,
            enforced_id=_enforced_id,
        )
        self.children.append(new_graph)

        return new_graph

    def add_group(
        self,
        name: str,
        children: Optional[List[GroupOrGraph]] = None,
        _enforced_id: int = None,
    ) -> "Group":

        new_group = Group(name, children or [], nml=self._nml, enforced_id=_enforced_id)  # type: ignore
        self.children.append(new_group)
        return new_group

    def get_total_node_count(self) -> int:
        return sum(len(graph.get_nodes()) for graph in self.flattened_graphs())

    def get_max_graph_id(self) -> int:
        return max((graph.id for graph in self.flattened_graphs()), default=0)

    def get_max_node_id(self) -> int:
        return max(
            (graph.get_max_node_id() for graph in self.flattened_graphs()),
            default=0,
        )

    def flattened_graphs(self) -> Generator["Graph", None, None]:
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

    def get_node_by_id(self, node_id: int) -> "Node":

        for graph in self.flattened_graphs():
            if graph.has_node_id(node_id):
                return graph.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def as_nml_group(self) -> "NmlGroup":  # type: ignore

        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self.children if isinstance(g, Group)],
        )


@attr.define()
class Node:
    position: Vector3
    _nml: "Skeleton"
    _id: int = attr.ib(init=False)
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
    _enforced_id: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._nml.element_id_generator.__next__()

    @property
    def id(self):
        return self._id


@attr.define()
class Graph:
    """
    Contains a collection of nodes and edges.
    """

    name: str
    _nml: "Skeleton"
    color: Optional[Vector4] = None
    _id: int = attr.ib(init=False)
    nx_graph: nx.Graph = attr.ib(init=False)
    group_id: Optional[int] = None

    _enforced_id: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        self.nx_graph = nx.Graph()
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._nml.element_id_generator.__next__()

    @property
    def id(self) -> int:
        return self._id

    def get_nodes(self) -> List[Node]:

        return [node_view[1] for node_view in self.nx_graph.nodes(data="obj")]

    def get_node_positions(self) -> np.ndarray:
        return np.array([node.position for node in self.get_nodes()])

    def get_node_by_id(self, node_id: int) -> Node:

        return self.nx_graph.nodes[node_id]["obj"]

    def has_node_id(self, node_id: int) -> bool:

        return node_id in self.nx_graph.nodes

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
        _enforced_id: Optional[int] = None,
        _nml: Optional["Skeleton"] = None,
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
            enforced_id=_enforced_id,
            nml=_nml or self._nml,
        )
        self.nx_graph.add_node(node.id, obj=node)
        return node

    def add_edge(self, node_1: Union[int, Node], node_2: Union[int, Node]) -> None:

        id_1 = node_1.id if isinstance(node_1, Node) else node_1
        id_2 = node_2.id if isinstance(node_2, Node) else node_2
        self.nx_graph.add_edge(id_1, id_2)

    def get_max_node_id(self) -> int:
        return max((node.id for node in self.get_nodes()), default=0)


@attr.define()
class Skeleton:
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
        self.time = int(str(self.time))  # typing: ignore

    def flattened_graphs(self) -> Generator["Graph", None, None]:

        return self.root_group.flattened_graphs()

    def get_graph_by_id(self, graph_id: int) -> Graph:

        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for graph in self.root_group.flattened_graphs():
            if graph.id == graph_id:
                return graph
        raise ValueError(f"No graph with id {graph_id} was found")

    def get_group_by_id(self, group_id: int) -> Group:

        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for group in self.root_group.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No group with id {group_id} was found")

    def add_graph(
        self,
        name: str,
        color: Optional[Vector4] = None,
        _nml: Optional["Skeleton"] = None,
        _enforced_id: Optional[int] = None,
    ) -> "Graph":

        return self.root_group.add_graph(
            name,
            color,
            _nml,
            _enforced_id,
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

    def get_node_by_id(self, node_id: int) -> Node:

        return self.root_group.get_node_by_id(node_id)

    @staticmethod
    def from_path(file_path: str) -> "Skeleton":

        with open(file_path, "rb") as file_handle:
            return Skeleton.from_nml(wknml.parse_nml(file_handle))

    @staticmethod
    def from_nml(nml: WkNML) -> "Skeleton":
        skeleton = Skeleton(
            name=nml.parameters.name,
            scale=nml.parameters.scale,
            offset=nml.parameters.offset,
            time=nml.parameters.time,
            edit_position=nml.parameters.editPosition,
            edit_rotation=nml.parameters.editRotation,
            zoom_level=nml.parameters.zoomLevel,
            task_bounding_box=nml.parameters.taskBoundingBox,
            user_bounding_boxes=nml.parameters.userBoundingBoxes,
        )

        groups_by_id = {}

        def visit_groups(nml_groups: List[NmlGroup], current_group: Group) -> None:

            for nml_group in nml_groups:
                sub_group = current_group.add_group(
                    name=nml_group.name, _enforced_id=nml_group.id
                )
                groups_by_id[sub_group.id] = sub_group
                visit_groups(nml_group.children, sub_group)

        visit_groups(nml.groups, skeleton.root_group)
        for nml_tree in nml.trees:
            if nml_tree.groupId is None:
                new_graph = skeleton.root_group.add_graph(
                    nml_tree.name, _enforced_id=nml_tree.id
                )
            else:
                new_graph = groups_by_id[nml_tree.groupId].add_graph(
                    nml_tree.name, _enforced_id=nml_tree.id
                )
            Skeleton.nml_tree_to_graph(new_graph, nml_tree)

        for comment in nml.comments:
            skeleton.get_node_by_id(comment.node).comment = comment.content

        for branchpoint in nml.branchpoints:
            node = skeleton.get_node_by_id(branchpoint.id)
            node.is_branchpoint = True
            if branchpoint.time != 0:
                node.branchpoint_time = branchpoint.time

        max_id = max(skeleton.get_max_graph_id(), skeleton.get_max_node_id())
        skeleton.element_id_generator = itertools.count(max_id + 1)

        return skeleton

    @staticmethod
    def nml_tree_to_graph(
        new_graph: "Graph",
        nml_tree: NmlTree,
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

        new_graph.color = nml_tree.color
        new_graph.name = nml_tree.name
        new_graph.group_id = nml_tree.groupId

        for nml_node in nml_tree.nodes:
            node_id = nml_node.id
            current_node = new_graph.add_node(
                position=nml_node.position,
                _enforced_id=node_id,
                radius=nml_node.radius,
            )

            for optional_attribute in optional_attribute_list:
                if getattr(nml_node, optional_attribute) is not None:
                    setattr(
                        current_node,
                        optional_attribute,
                        getattr(nml_node, optional_attribute),
                    )

        for edge in nml_tree.edges:
            new_graph.add_edge(edge.source, edge.target)

        return new_graph

    def write(self, out_path: str) -> None:

        nml = NMLExporter.generate_nml(
            self.root_group,
            self._get_nml_parameters(),
        )

        with open(out_path, "wb") as f:
            wknml.write_nml(f, nml)

    def _get_nml_parameters(self) -> Dict[str, Any]:

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

import itertools
from os import PathLike
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import attr
import networkx as nx
import numpy as np

import webknossos.skeleton.nml as wknml
from webknossos.skeleton.nml.from_skeleton import from_skeleton as nml_from_skeleton
from webknossos.skeleton.nml.to_skeleton import to_skeleton as nml_to_skeleton

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union["Group", "Graph"]


@attr.define()
class Group:
    _id: int = attr.ib(init=False)
    name: str
    _children: List[GroupOrGraph]
    _nml: "Skeleton"
    is_root_group: bool = False
    _enforced_id: Optional[int] = None

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._nml.element_id_generator.__next__()

    @property
    def id(self) -> int:
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
        self._children.append(new_graph)

        return new_graph

    @property
    def children(self) -> Iterator[GroupOrGraph]:
        return (child for child in self._children)

    def add_group(
        self,
        name: str,
        children: Optional[List[GroupOrGraph]] = None,
        _enforced_id: int = None,
    ) -> "Group":

        new_group = Group(name, children or [], nml=self._nml, enforced_id=_enforced_id)
        self._children.append(new_group)
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
        for child in self._children:
            if isinstance(child, Group):
                yield from child.flattened_graphs()
            else:
                yield child

    def flattened_groups(self) -> Generator["Group", None, None]:
        for child in self._children:
            if isinstance(child, Group):
                yield child
                yield from child.flattened_groups()

    def get_node_by_id(self, node_id: int) -> "Node":
        for graph in self.flattened_graphs():
            if graph.has_node_id(node_id):
                return graph.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def as_nml_group(self) -> wknml.Group:
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._children if isinstance(g, Group)],
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
    def id(self) -> int:
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
    def from_path(file_path: PathLike) -> "Skeleton":
        with open(file_path, "rb") as file_handle:
            return nml_to_skeleton(wknml.parse_nml(file_handle))

    def write(self, out_path: PathLike) -> None:
        nml = nml_from_skeleton(
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

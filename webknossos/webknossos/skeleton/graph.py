from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import attr
import networkx as nx
import numpy as np

from .node import Node

if TYPE_CHECKING:
    from webknossos.skeleton import Group, Skeleton

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


@attr.define(kw_only=True)
class Graph:
    """
    Contains a collection of nodes and edges.
    """

    name: str
    group: "Group" = attr.ib(eq=False, repr=False)
    _nml: "Skeleton" = attr.ib(eq=False, repr=False)
    _id: int = attr.ib(init=False)
    nx_graph: nx.Graph = attr.ib(
        init=False,
        repr=lambda nx_graph: f"<n={len(nx_graph.nodes)},e={len(nx_graph.edges)}>",
        eq=lambda nx_graph: (
            sorted(nx_graph.nodes(data="obj")),
            sorted(nx_graph.edges),
        ),
    )
    color: Optional[Vector4] = None
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)

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

    def __hash__(self) -> int:
        return self._id

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Union

import networkx as nx
import numpy as np

from .node import Node

if TYPE_CHECKING:
    from webknossos.skeleton import Group, Skeleton

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


def _get_id(node_or_id: Union[Node, int]) -> int:
    if isinstance(node_or_id, Node):
        return node_or_id.id
    else:
        return node_or_id


class _AdjDict(MutableMapping):
    def __init__(self, *, node_dict: "_NodeDict") -> None:
        self._id_to_attrs: Dict[int, Any] = {}
        self._node_dict = node_dict

    def __getitem__(self, key: Union[Node, int]) -> Any:
        return self._id_to_attrs[_get_id(key)]

    def __setitem__(self, key: Node, value: Dict) -> None:
        self._id_to_attrs[_get_id(key)] = value

    def __delitem__(self, key: Union[Node, int]) -> None:
        del self._id_to_attrs[_get_id(key)]

    def __iter__(self) -> Iterator[Node]:
        return (self._node_dict.get_node(i) for i in self._id_to_attrs)

    def __len__(self) -> int:
        return len(self._id_to_attrs)


class _NodeDict(MutableMapping):
    def __init__(self) -> None:
        self._id_to_attrs: Dict[int, Any] = {}
        self._id_to_node: Dict[int, Node] = {}

    def __getitem__(self, key: Union[Node, int]) -> Any:
        return self._id_to_attrs[_get_id(key)]

    def __setitem__(self, key: Node, value: Dict) -> None:
        self._id_to_node[key.id] = key
        self._id_to_attrs[key.id] = value

    def __delitem__(self, key: Union[Node, int]) -> None:
        del self._id_to_node[_get_id(key)]
        del self._id_to_attrs[_get_id(key)]

    def __iter__(self) -> Iterator[Node]:
        return iter(self._id_to_node.values())

    def __len__(self) -> int:
        return len(self._id_to_attrs)

    def get_node(self, id_: int) -> Node:
        return self._id_to_node[id_]


class Graph(nx.Graph):
    """
    Contains a collection of nodes and edges.
    This class inherits from [`networkx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html).
    For further methods, please [check the networkx documentation](https://networkx.org/documentation/stable/reference/classes/graph.html#methods).
    """

    def __init__(
        self,
        name: str,
        group: "Group",
        skeleton: "Skeleton",
        color: Optional[Vector4] = None,
        enforced_id: Optional[int] = None,
    ) -> None:
        self.node_dict_factory = _NodeDict
        self.adjlist_outer_dict_factory = lambda: _AdjDict(node_dict=self._node)
        self.adjlist_inner_dict_factory = lambda: _AdjDict(node_dict=self._node)

        super().__init__()

        self.name = name
        self.group = group
        self.color = color

        # read-only member, exposed via properties
        if enforced_id is not None:
            self._id = enforced_id
        else:
            self._id = skeleton.element_id_generator.__next__()

        # only used internally
        self._skeleton = skeleton

    def __eq__(self, o: object) -> bool:
        get_comparable = lambda graph: (
            graph.name,
            graph.id,
            graph.color,
            sorted(graph.nodes),
            sorted(graph.edges),
        )
        return get_comparable(self) == get_comparable(o)

    @property
    def id(self) -> int:
        return self._id

    def get_node_positions(self) -> np.ndarray:
        return np.array([node.position for node in self.nodes])

    def get_node_by_id(self, node_id: int) -> Node:
        return self._node.get_node(node_id)

    def add_node(  # pylint: disable=arguments-differ
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
            skeleton=self._skeleton,
        )
        super().add_node(node)
        return node

    def get_max_node_id(self) -> int:
        return max((node.id for node in self.nodes), default=0)

    def __hash__(self) -> int:
        return self._id

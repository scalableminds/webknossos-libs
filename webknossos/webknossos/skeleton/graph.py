from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Union

import networkx as nx
import numpy as np

from webknossos.geometry import Vec3Int, Vec3IntLike

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


class _NodeDict(MutableMapping):
    """Dict-like container for the `Graph` class below to handle both nodes and ids as keys.
    Only when setting the value of a node for the first time the actual node must be passed.
    This allows to keep a reference from all ids to the nodes, which can be looked up
    using `get_node()`. Iterating over the keys yields the `Node` objects."""

    def __init__(self) -> None:
        self._id_to_attrs: Dict[int, Any] = {}
        self._id_to_node: Dict[int, Node] = {}

    def __getitem__(self, key: Union[Node, int]) -> Any:
        return self._id_to_attrs[_get_id(key)]

    def __setitem__(self, key: Union[Node, int], value: Dict) -> None:
        key_id = _get_id(key)
        if key_id not in self._id_to_node:
            if isinstance(key, Node):
                self._id_to_node[key_id] = key
            else:
                raise ValueError(
                    f"Tried to add node {key}, which does not exist yet, to a graph."
                    + "For insertion, the node must be an instance of the Node class."
                )
        self._id_to_attrs[key_id] = value

    def __delitem__(self, key: Union[Node, int]) -> None:
        del self._id_to_node[_get_id(key)]
        del self._id_to_attrs[_get_id(key)]

    def __iter__(self) -> Iterator[Node]:
        return iter(self._id_to_node.values())

    def __len__(self) -> int:
        return len(self._id_to_attrs)

    def get_node(self, id_: int) -> Node:
        return self._id_to_node[id_]


class _AdjDict(MutableMapping):
    """Dict-like container for the `Graph` class below to handle both nodes and ids as keys.
    Needs a reference to the _node attribute (of class `_NodeDict`) of the graph object
    to get a reference from ids to nodes. Iterating over the keys yields the `Node` objects.

    See Graph.__init__ for more details"""

    def __init__(self, *, node_dict: _NodeDict) -> None:
        self._id_to_attrs: Dict[int, Any] = {}
        self._node_dict = node_dict

    def __getitem__(self, key: Union[Node, int]) -> Any:
        return self._id_to_attrs[_get_id(key)]

    def __setitem__(self, key: Union[Node, int], value: Dict) -> None:
        self._id_to_attrs[_get_id(key)] = value

    def __delitem__(self, key: Union[Node, int]) -> None:
        del self._id_to_attrs[_get_id(key)]

    def __iter__(self) -> Iterator[Node]:
        return (self._node_dict.get_node(i) for i in self._id_to_attrs)

    def __len__(self) -> int:
        return len(self._id_to_attrs)


class Graph(nx.Graph):
    """
    Contains a collection of nodes and edges.
    This class inherits from [`networkx.Graph`](https://networkx.org/documentation/stable/reference/classes/graph.html).
    For further methods, please [check the networkx documentation](https://networkx.org/documentation/stable/reference/classes/graph.html#methods).

    See Graph.__init__ for more details.

    A small usage example:

    ```python
    graph = skeleton.add_graph("a graph")
    node_1 = graph.add_node(position=(0, 0, 0), comment="node 1")
    node_2 = graph.add_node(position=(100, 100, 100), comment="node 2")

    graph.add_edge(node_1, node_2)
    ```
    """

    def __init__(
        self,
        name: str,
        group: "Group",
        skeleton: "Skeleton",
        color: Optional[Vector4] = None,
        enforced_id: Optional[int] = None,
    ) -> None:
        """
        To create a graph, it is recommended to use `Skeleton.add_graph` or
        `Group.add_graph`. That way, the newly created graph is automatically
        attached as a child to the object the method was called on.
        """

        # To be able to reference nodes by id after adding them for the first time, we use custom dict-like classes
        # for the networkx-graph structures, that have nodes as keys:
        # * `self._node`: _NodeDict
        #   holding the attributes of nodes, keeping references from ids to nodes
        # * `self._adj`: _AdjDict on the first two levels
        #   holding edge attributes on the last level, using self._node to convert ids to nodes
        #
        # It's important to set the attributes before the parent's init so that they shadow the class-attributes.
        #
        # For further details, see the *Subclasses* section here: https://networkx.org/documentation/stable/reference/classes/graph.html

        self.node_dict_factory = _NodeDict
        # The lambda works because self._node is set before self._adj in networkx.Graph.__init__
        # and because the lambda is evaluated lazily.
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
            self._id = skeleton._element_id_generator.__next__()

        # only used internally
        self._skeleton = skeleton

    def __to_tuple_for_comparison(self) -> Tuple:
        return (
            self.name,
            self.id,
            self.color,
            sorted(self.nodes),
            sorted(self.edges),
        )

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Graph) and (
            self.__to_tuple_for_comparison() == o.__to_tuple_for_comparison()
        )

    @property
    def id(self) -> int:
        """Read-only property."""
        return self._id

    def get_node_positions(self) -> np.ndarray:
        """Returns an numpy array with the positions of all nodes of this graph."""
        return np.array([node.position for node in self.nodes])

    def get_node_by_id(self, node_id: int) -> Node:
        """Returns the node in this graph with the requested id."""
        return self._node.get_node(node_id)

    def add_node(  # pylint: disable=arguments-differ
        self,
        position: Vec3IntLike,
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
        """
        Adds a node to the graph. Apart from the mandatory `position` parameter,
        there are several optional parameters which can be used to encode
        additional information. For example, the comment will be shown by the
        webKnossos UI.
        """
        node = Node(
            position=Vec3Int(position),
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
        """Returns the highest node id."""
        return max((node.id for node in self.nodes), default=0)

    def __hash__(self) -> int:
        return self._id

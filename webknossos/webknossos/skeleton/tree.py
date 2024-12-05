from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Union

import networkx as nx
import numpy as np

from ..geometry import Vec3Int, Vec3IntLike
from ..utils import warn_deprecated
from .node import Node

if TYPE_CHECKING:
    from .group import Group
    from .skeleton import Skeleton

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


def _get_id(node_or_id: Union[Node, int]) -> int:
    if isinstance(node_or_id, Node):
        return node_or_id.id
    else:
        return node_or_id


class _NodeDict(MutableMapping):
    """Dict-like container for the `Tree` class below to handle both nodes and ids as keys.
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
                    f"Tried to add node {key}, which does not exist yet, to a tree."
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
    """Dict-like container for the `Tree` class below to handle both nodes and ids as keys.
    Needs a reference to the _node attribute (of class `_NodeDict`) of the graph object
    to get a reference from ids to nodes. Iterating over the keys yields the `Node` objects.

    See Tree.__init__ for more details"""

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


class Tree(nx.Graph):
    """Contains a collection of nodes and edges in a graph structure.

    Despite the name, trees may contain cycles. This class inherits from networkx.Graph
    and provides additional functionality specific to neural annotation tasks.

    Args:
        name (str): The name of the tree.
        group (Group): The group this tree belongs to.
        skeleton (Skeleton): The skeleton this tree is part of.
        color (Optional[Vector4], optional): RGBA color values for the tree. Defaults to None.
        enforced_id (Optional[int], optional): Specific ID to use for the tree. Defaults to None.

    Returns:
        Tree: A new Tree instance that represents a collection of nodes and edges.

    Raises:
        ValueError: If the tree name is empty or if the group or skeleton is None.
        TypeError: If the color value is not a valid Vector4 type when provided.

    Note:
        It is recommended to create trees using `Skeleton.add_tree` or `Group.add_tree`
        instead of instantiating this class directly. This ensures proper parent-child
        relationships are maintained.

    Examples:
        Create a new tree with nodes and edges:
        ```python
        # First create a skeleton (parent object)
        skeleton = Skeleton("example_skeleton")

        # Add a new tree to the skeleton
        tree = skeleton.add_tree("dendrite_1")

        # Add nodes with 3D positions
        soma = tree.add_node(position=(0, 0, 0), comment="soma")
        branch1 = tree.add_node(position=(100, 0, 0), comment="branch1")
        branch2 = tree.add_node(position=(0, 100, 0), comment="branch2")

        # Connect nodes with edges
        tree.add_edge(soma, branch1)
        tree.add_edge(soma, branch2)

        # Access node positions
        positions = tree.get_node_positions()  # Returns numpy array of all positions
        ```

    For additional graph operations, see the
    [networkx documentation](https://networkx.org/documentation/stable/reference/classes/graph.html#methods).
    """

    def __init__(
        self,
        name: str,
        group: "Group",
        skeleton: "Skeleton",
        color: Optional[Vector4] = None,
        enforced_id: Optional[int] = None,  # noqa: ARG002 Unused method argument: `enforced_id`
    ) -> None:
        """
        To create a tree, it is recommended to use `Skeleton.add_tree` or
        `Group.add_tree`. That way, the newly created tree is automatically
        attached as a child to the object the method was called on.
        """

        super().__init__()
        # Note that id is set up in __new__
        self.name = name
        self.group = group
        self.color = color

        # only used internally
        self._skeleton = skeleton

    def __new__(
        cls,
        name: str,  # noqa: ARG003 Unused class method argument: `name`
        group: "Group",  # noqa: ARG003 Unused class method argument: `group`
        skeleton: "Skeleton",
        color: Optional[Vector4] = None,  # noqa: ARG003 Unused class method argument: `color`
        enforced_id: Optional[int] = None,
    ) -> "Tree":
        self = super().__new__(cls)

        # self._id is a read-only member, exposed via properties.
        # It is set in __new__ instead of __init__ so that pickling/unpickling
        # works without problems. As long as the deserialization of a tree instance
        # is not finished, the object is only half-initialized. Since self._id
        # is needed by __hash__, an error would be raised otherwise.
        # Also see:
        # https://stackoverflow.com/questions/46283738/attributeerror-when-using-python-deepcopy
        if enforced_id is not None:
            self._id = enforced_id
        else:
            self._id = skeleton._element_id_generator.__next__()

        return self

    def __getnewargs__(self) -> Tuple:
        # pickle.dump will pickle instances of Tree so that the following
        # tuple is passed as arguments to __new__.
        return (self.name, self.group, self._skeleton, self.color, self._id)

    # node_dict_factory, adjlist_outer_dict_factory and adjlist_inner_dict_factory are used by networkx
    # from which we subclass.
    # To be able to reference nodes by id after adding them for the first time, we use custom dict-like classes
    # for the networkx-graph structures, that have nodes as keys:
    #     * `self._node`: _NodeDict
    #        holding the attributes of nodes, keeping references from ids to nodes
    #     * `self._adj`: _AdjDict on the first two levels
    #       holding edge attributes on the last level, using self._node to convert ids to nodes
    # It's important to set the attributes before the parent's init so that they shadow the class-attributes.
    # For further details, see the *Subclasses* section here: https://networkx.org/documentation/stable/reference/classes/graph.html

    node_dict_factory = _NodeDict

    def adjlist_outer_dict_factory(self) -> _AdjDict:
        # self._node will already be available when this method is called, because networkx.Graph.__init__
        # sets up the nodes first and then the edges (i.e., adjacency list).
        return _AdjDict(node_dict=self._node)

    def adjlist_inner_dict_factory(self) -> _AdjDict:
        return _AdjDict(node_dict=self._node)

    def __to_tuple_for_comparison(self) -> Tuple:
        return (
            self.name,
            self.id,
            self.color,
            sorted(self.nodes),
            sorted(self.edges),
        )

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Tree) and (
            self.__to_tuple_for_comparison() == o.__to_tuple_for_comparison()
        )

    @property
    def id(self) -> int:
        """The unique identifier of the tree.

        Returns:
            int: A unique identifier for this tree instance.

        Note:
            This is a read-only property that is set during tree creation.
        """
        return self._id

    def get_node_positions(self) -> np.ndarray:
        """Get positions of all nodes in the tree.

        Returns:
            np.ndarray: A numpy array of shape (N, 3) containing the 3D positions
                of all N nodes in the tree. Each row represents a node's (x, y, z)
                coordinates.
        """
        return np.array([node.position for node in self.nodes])

    def get_node_by_id(self, node_id: int) -> Node:
        """Retrieve a node using its unique identifier.

        Args:
            node_id (int): The unique identifier of the node to retrieve.

        Returns:
            Node: The node with the specified ID.

        Raises:
            KeyError: If no node exists with the given ID in this tree.
        """
        return self._node.get_node(node_id)

    def add_node(
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
        """Add a new node to the tree.

        Creates a new node at the specified position and adds it to the tree.

        Args:
            position (Vec3IntLike): The 3D coordinates (x, y, z) of the node.
            comment (Optional[str], optional): A text comment associated with the node.
                Visible in the WEBKNOSSOS UI. Defaults to None.
            radius (Optional[float], optional): Node radius for visualization.
                Defaults to None.
            rotation (Optional[Vector3], optional): 3D rotation vector for the node.
                Defaults to None.
            inVp (Optional[int], optional): Viewport information. Defaults to None.
            inMag (Optional[int], optional): Magnification level. Defaults to None.
            bitDepth (Optional[int], optional): Bit depth for node data.
                Defaults to None.
            interpolation (Optional[bool], optional): Whether to use interpolation.
                Defaults to None.
            time (Optional[int], optional): Timestamp for the node. Defaults to None.
            is_branchpoint (bool, optional): Whether this node is a branch point.
                Defaults to False.
            branchpoint_time (Optional[int], optional): Timestamp for branch point
                creation. Defaults to None.
            _enforced_id (Optional[int], optional): Internal use only. Forces a
                specific node ID. Defaults to None.

        Returns:
            Node: The newly created and added node.

        Examples:
            ```python
            # Add a simple node with just position
            node1 = tree.add_node(position=(100, 200, 300))

            # Add a node with additional properties
            node2 = tree.add_node(
                position=(150, 250, 350),
                comment="Dendrite branch point",
                radius=2.5,
                is_branchpoint=True
            )
            ```
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
        """Get the highest node ID in the tree.

        Returns:
            int: The maximum node ID present in the tree. Returns 0 if the tree
                has no nodes.
        """
        return max((node.id for node in self.nodes), default=0)

    def __hash__(self) -> int:
        return self._id


class Graph(Tree):
    """Deprecated, please use `Tree` instead."""

    def __init__(
        self,
        name: str,
        group: "Group",
        skeleton: "Skeleton",
        color: Optional[Vector4] = None,
        enforced_id: Optional[int] = None,
    ) -> None:
        warn_deprecated("Graph", "Tree")
        super().__init__(
            name=name,
            group=group,
            skeleton=skeleton,
            color=color,
            enforced_id=enforced_id,
        )

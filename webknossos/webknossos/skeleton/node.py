from typing import TYPE_CHECKING, Optional, Tuple

import attr

from webknossos.geometry import Vec3Int

if TYPE_CHECKING:
    from webknossos.skeleton import Skeleton

Vec3Float = Tuple[float, float, float]


# Defining an order on nodes is necessary to allow to sort them,
# which is used in the graph's equality check, see Graph.__eq__().
@attr.define(order=True)
class Node:
    position: Vec3Int
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False, order=False)
    _id: int = attr.ib(init=False)
    comment: Optional[str] = None
    radius: Optional[float] = None
    rotation: Optional[Vec3Float] = None
    inVp: Optional[int] = None
    inMag: Optional[int] = None
    bitDepth: Optional[int] = None
    interpolation: Optional[bool] = None
    time: Optional[int] = None

    is_branchpoint: bool = False
    branchpoint_time: Optional[int] = None
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)

    @classmethod
    def _set_init_docstring(cls) -> None:
        Node.__init__.__doc__ = """
        To create a node, it is recommended to use `Graph.add_node`. That way,
        the newly created group is automatically attached as a child to the
        graph.

        A small usage example:

        ```python
        graph = skeleton.add_graph("a graph")
        node_1 = graph.add_node(position=(0, 0, 0), comment="node 1")
        node_2 = graph.add_node(position=(100, 100, 100), comment="node 2")

        graph.add_edge(node_1, node_2)
        ```
        """

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._skeleton._element_id_generator.__next__()

    @property
    def id(self) -> int:
        return self._id

    def __hash__(self) -> int:
        return self._id


Node._set_init_docstring()

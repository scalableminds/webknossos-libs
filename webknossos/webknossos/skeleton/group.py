from typing import TYPE_CHECKING, Generator, Iterator, Optional, Set, Tuple, Union, cast

import attr
from boltons.strutils import unit_len

import webknossos.skeleton.nml as wknml

from .graph import Graph

if TYPE_CHECKING:
    from webknossos.skeleton import Node, Skeleton


Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
GroupOrGraph = Union["Group", Graph]


@attr.define()
class Group:
    _id: int = attr.ib(init=False)
    name: str
    _children: Set[GroupOrGraph] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'children')}>",
    )
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False)
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._skeleton.element_id_generator.__next__()

    @property
    def id(self) -> int:
        return self._id

    def add_graph(
        self,
        name: str,
        color: Optional[Union[Vector4, Vector3]] = None,
        _enforced_id: Optional[int] = None,
    ) -> Graph:
        if color is not None and len(color) == 3:
            color = cast(Optional[Vector4], color + (1.0,))
        color = cast(Optional[Vector4], color)
        new_graph = Graph(
            name=name,
            color=color,
            group=self,
            skeleton=self._skeleton,
            enforced_id=_enforced_id,
        )
        self._children.add(new_graph)

        return new_graph

    @property
    def children(self) -> Iterator[GroupOrGraph]:
        return (child for child in self._children)

    def add_group(
        self,
        name: str,
        _enforced_id: Optional[int] = None,
    ) -> "Group":
        new_group = Group(name, skeleton=self._skeleton, enforced_id=_enforced_id)
        self._children.add(new_group)
        return new_group

    def get_total_node_count(self) -> int:
        return sum(graph.number_of_nodes() for graph in self.flattened_graphs())

    def get_max_graph_id(self) -> int:
        return max((graph.id for graph in self.flattened_graphs()), default=0)

    def get_max_node_id(self) -> int:
        return max(
            (graph.get_max_node_id() for graph in self.flattened_graphs()),
            default=0,
        )

    def flattened_graphs(self) -> Generator[Graph, None, None]:
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
            if graph.has_node(node_id):
                return graph.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def get_graph_by_id(self, graph_id: int) -> Graph:
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for graph in self.flattened_graphs():
            if graph.id == graph_id:
                return graph
        raise ValueError(f"No graph with id {graph_id} was found")

    def get_group_by_id(self, group_id: int) -> "Group":
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for group in self.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No group with id {group_id} was found")

    def as_nml_group(self) -> wknml.Group:
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._children if isinstance(g, Group)],
        )

    def __hash__(self) -> int:
        return self._id

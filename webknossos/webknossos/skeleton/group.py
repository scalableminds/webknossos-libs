from typing import TYPE_CHECKING, Generator, Iterator, List, Optional, Tuple, Union

import attr

import webknossos.skeleton.nml as wknml

from .graph import Graph

if TYPE_CHECKING:
    from webknossos.skeleton import Node, Skeleton


Vector4 = Tuple[float, float, float, float]
GroupOrGraph = Union["Group", Graph]


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
    ) -> Graph:

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
            if graph.has_node_id(node_id):
                return graph.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def as_nml_group(self) -> wknml.Group:
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._children if isinstance(g, Group)],
        )

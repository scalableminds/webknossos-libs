from typing import TYPE_CHECKING, Iterator, Optional, Set, Tuple, Union, cast

import attr
from boltons.strutils import unit_len

import webknossos._nml as wknml
from webknossos.utils import warn_deprecated

from .tree import Tree

if TYPE_CHECKING:
    from webknossos.skeleton import Node, Skeleton


Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
GroupOrTree = Union["Group", Tree]


@attr.define()
class Group:
    _id: int = attr.ib(init=False)
    name: str
    _child_groups: Set["Group"] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'child group')}>",
    )
    _child_trees: Set[Tree] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'child tree')}>",
    )
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False)
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)

    @classmethod
    def _set_init_docstring(cls) -> None:
        Group.__init__.__doc__ = """
        To create a group, it is recommended to use `webknossos.skeleton.skeleton.Skeleton.add_group` or
        `Group.add_group`. That way, the newly created group
        is automatically attached as a child to the object the method was
        called on.

        A small usage example:

        ```python
        subgroup = group.add_group("a subgroup")
        tree = subgroup.add_tree("a tree")
        ```
        """

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._skeleton._element_id_generator.__next__()

    @property
    def id(self) -> int:
        """Read-only property."""
        return self._id

    def add_tree(
        self,
        name: str,
        color: Optional[Union[Vector4, Vector3]] = None,
        _enforced_id: Optional[int] = None,
    ) -> Tree:
        """Adds a tree to the current group with the provided name (and color if specified)."""

        if color is not None and len(color) == 3:
            color = cast(Optional[Vector4], color + (1.0,))
        color = cast(Optional[Vector4], color)
        new_tree = Tree(
            name=name,
            color=color,
            group=self,
            skeleton=self._skeleton,
            enforced_id=_enforced_id,
        )
        self._child_trees.add(new_tree)

        return new_tree

    def add_graph(
        self,
        name: str,
        color: Optional[Union[Vector4, Vector3]] = None,
        _enforced_id: Optional[int] = None,
    ) -> Tree:
        """Deprecated, please use `add_tree`."""
        warn_deprecated("add_graph()", "add_tree()")
        return self.add_tree(name=name, color=color, _enforced_id=_enforced_id)

    @property
    def children(self) -> Iterator[GroupOrTree]:
        """Returns all (immediate) children (groups and trees) as an iterator."""
        yield from self.trees
        yield from self.groups

    @property
    def trees(self) -> Iterator[Tree]:
        """Returns all (immediate) tree children as an iterator.
        Use flattened_trees if you need also need trees within subgroups."""
        return (child for child in self._child_trees)

    @property
    def graphs(self) -> Iterator[Tree]:
        """Deprecated, please use `trees`."""
        warn_deprecated("graphs", "trees")
        return self.trees

    @property
    def groups(self) -> Iterator["Group"]:
        """Returns all (immediate) group children as an iterator.
        Use flattened_groups if you need also need groups within subgroups."""
        return (child for child in self._child_groups)

    def add_group(
        self,
        name: str,
        _enforced_id: Optional[int] = None,
    ) -> "Group":
        """Adds a (sub) group to the current group with the provided name."""
        new_group = Group(name, skeleton=self._skeleton, enforced_id=_enforced_id)
        self._child_groups.add(new_group)
        return new_group

    def get_total_node_count(self) -> int:
        """Returns the total number of nodes of all trees within this group (and its subgroups)."""
        return sum(tree.number_of_nodes() for tree in self.flattened_trees())

    def get_max_tree_id(self) -> int:
        """Returns the highest tree id of all trees within this group (and its subgroups)."""
        return max((tree.id for tree in self.flattened_trees()), default=0)

    def get_max_graph_id(self) -> int:
        """Deprecated, please use `get_max_tree_id`."""
        warn_deprecated("get_max_graph_id()", "get_max_tree_id()")
        return self.get_max_tree_id()

    def get_max_node_id(self) -> int:
        """Returns the highest node id of all nodes of all trees within this group (and its subgroups)."""
        return max(
            (tree.get_max_node_id() for tree in self.flattened_trees()),
            default=0,
        )

    def flattened_trees(self) -> Iterator[Tree]:
        """Returns an iterator of all trees within this group (and its subgroups)."""
        yield from self.trees
        for group in self.groups:
            yield from group.flattened_trees()

    def flattened_graphs(self) -> Iterator[Tree]:
        """Deprecated, please use `flattened_trees`."""
        warn_deprecated("flattened_graphs()", "flattened_trees()")
        return self.flattened_trees()

    def flattened_groups(self) -> Iterator["Group"]:
        """Returns an iterator of all groups within this group (and its subgroups)."""
        for group in self.groups:
            yield group
            yield from group.flattened_groups()

    def get_node_by_id(self, node_id: int) -> "Node":
        """Returns the node which has the specified node id."""
        for tree in self.flattened_trees():
            if tree.has_node(node_id):
                return tree.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def get_tree_by_id(self, tree_id: int) -> Tree:
        """Returns the tree which has the specified tree id."""
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for tree in self.flattened_trees():
            if tree.id == tree_id:
                return tree
        raise ValueError(f"No tree with id {tree_id} was found")

    def get_graph_by_id(self, graph_id: int) -> Tree:
        """Deprecated, please use `get_tree_by_id`."""
        warn_deprecated("get_graph_by_id()", "get_tree_by_id()")
        return self.get_tree_by_id(graph_id)

    def get_group_by_id(self, group_id: int) -> "Group":
        """Returns the group which has the specified group id."""
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for group in self.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No group with id {group_id} was found")

    def as_nml_group(self) -> wknml.Group:
        """Returns a named tuple representation of this group."""
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._child_groups],
        )

    def __hash__(self) -> int:
        return self._id


Group._set_init_docstring()

import copy
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union, cast

import attr
from boltons.strutils import unit_len

import webknossos._nml as wknml

from .tree import Tree

if TYPE_CHECKING:
    from .node import Node
    from .skeleton import Skeleton


Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
GroupOrTree = Union["Group", Tree]


@attr.define()
class Group:
    """A hierarchical container for organizing trees and subgroups in WEBKNOSSOS skeleton annotations.

    The Group class provides a way to organize skeleton trees into logical collections,
    supporting nested hierarchies of groups and trees. Groups can contain both trees
    (collections of connected nodes) and other groups, enabling flexible organization
    of complex annotations.

    Attributes:
        name: Name of the group.
        id: Read-only unique identifier for the group.
        children: Iterator over all immediate children (both groups and trees).
        groups: Iterator over immediate subgroups.
        trees: Iterator over immediate trees.

    The group hierarchy supports:
        - Nested organization (groups within groups)
        - Tree management (adding, removing, retrieving trees)
        - Node access across all contained trees
        - Flattened views of the hierarchy

    Examples:
        Create a hierarchical structure:
        ```python
        # Create groups for different neuron parts
        neuron = skeleton.add_group("neuron_42")
        dendrites = neuron.add_group("dendrites")
        axons = neuron.add_group("axons")

        # Add trees to groups
        basal = dendrites.add_tree("basal_dendrite")
        apical = dendrites.add_tree("apical_dendrite", color=(1, 0, 0, 1))
        axon_tree = axons.add_tree("main_axon")

        # Work with the hierarchy
        print(f"Total nodes: {neuron.get_total_node_count()}")
        for tree in dendrites.trees:
            print(f"Dendrite tree: {tree.name}")

        # Access nodes across all trees
        node = neuron.get_node_by_id(123)
        ```

        Copy existing trees:
        ```python
        # Copy a tree to another group
        template = existing_group.get_tree_by_id(42)
        copy = new_group.add_tree(template, color=(0, 1, 0, 1))
        ```

    Notes:
        - Groups maintain unique IDs for all contained elements
        - Use flattened_* methods to access nested elements recursively
        - Trees should be added using add_tree rather than created directly
        - The group hierarchy is part of a Skeleton instance

    See Also:
        - Tree: Class representing connected node structures
        - Node: Class representing individual 3D points
        - Skeleton: Root container for all annotation data
    """

    _id: int = attr.ib(init=False)
    name: str
    _child_groups: set["Group"] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'child group')}>",
    )
    _child_trees: set[Tree] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'child tree')}>",
    )
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False)
    _enforced_id: int | None = attr.ib(None, eq=False, repr=False)

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
        name_or_tree: str | Tree,
        color: Vector4 | Vector3 | None = None,
        _enforced_id: int | None = None,
        metadata: dict[str, str | int | float | Sequence[str]] = {},
    ) -> Tree:
        """Adds a new tree or copies an existing tree to this group.

        This method supports two ways of adding trees:
        1. Creating a new tree by providing a name
        2. Copying an existing tree from another location

        Args:
            name_or_tree: Either a string name for a new tree or an existing Tree instance to copy.
            color: Optional RGBA color tuple (r, g, b, a) or RGB tuple (r, g, b).
                  If an RGB tuple is provided, alpha will be set to 1.0.
            _enforced_id: Optional specific ID for the tree (internal use).

        Returns:
            Tree: The newly created or copied tree.

        Examples:
            ```python
            # Create new tree
            tree = group.add_tree("dendrite_1", color=(1, 0, 0, 1))

            # Copy existing tree
            copy = group.add_tree(existing_tree)
            ```

        Notes:
            When copying a tree, a new ID will be generated if the original ID
            already exists in this group.
        """
        if color is not None and len(color) == 3:
            color = cast(Vector4 | None, color + (1.0,))
        color = cast(Vector4 | None, color)

        if isinstance(name_or_tree, str):
            name = name_or_tree
            new_tree = Tree(
                name=name,
                color=color,
                group=self,
                skeleton=self._skeleton,
                enforced_id=_enforced_id,
                metadata=metadata,
            )
            self._child_trees.add(new_tree)

            return new_tree
        else:
            tree = cast(Tree, name_or_tree)
            new_tree = copy.deepcopy(tree)

            if color is not None:
                new_tree.color = color

            if _enforced_id is not None:
                assert not self.has_tree_id(_enforced_id), (
                    "A tree with the specified _enforced_id already exists in this group."
                )
                new_tree._id = _enforced_id

            if self.has_tree_id(tree.id):
                new_tree._id = self._skeleton._element_id_generator.__next__()

            new_tree.group = self
            new_tree.skeleton = self._skeleton

            self._child_trees.add(new_tree)
            return new_tree

    def remove_tree_by_id(self, tree_id: int) -> None:
        self._child_trees.remove(self.get_tree_by_id(tree_id))

    @property
    def children(self) -> Iterator[GroupOrTree]:
        """Returns an iterator over all immediate children (groups and trees).

        This property provides access to both groups and trees that are direct
        children of this group, in no particular order. For nested access,
        use flattened_trees() or flattened_groups().

        Returns:
            Iterator[GroupOrTree]: Iterator yielding all immediate child groups and trees.

        Examples:
            ```python
            # Print all immediate children
            for child in group.children:
                if isinstance(child, Tree):
                    print(f"Tree: {child.name}")
                else:
                    print(f"Group: {child.name}")
            ```
        """
        yield from self.trees
        yield from self.groups

    @property
    def trees(self) -> Iterator[Tree]:
        """Returns all (immediate) tree children as an iterator.
        Use flattened_trees if you also need trees within subgroups."""
        return (child for child in self._child_trees)

    @property
    def groups(self) -> Iterator["Group"]:
        """Returns all (immediate) group children as an iterator.
        Use flattened_groups if you need also need groups within subgroups."""
        return (child for child in self._child_groups)

    def add_group(
        self,
        name: str,
        _enforced_id: int | None = None,
    ) -> "Group":
        """Creates and adds a new subgroup to this group.

        Args:
            name: Name for the new group.
            _enforced_id: Optional specific ID for the group (internal use).

        Returns:
            Group: The newly created group.

        Examples:
            ```python
            # Create nested group hierarchy
            dendrites = neuron.add_group("dendrites")
            basal = dendrites.add_group("basal")
            apical = dendrites.add_group("apical")
            ```
        """
        new_group = Group(name, skeleton=self._skeleton, enforced_id=_enforced_id)
        self._child_groups.add(new_group)
        return new_group

    def get_total_node_count(self) -> int:
        """Counts all nodes in all trees within this group and its subgroups.

        Returns:
            int: Total number of nodes across all contained trees.

        Examples:
            ```python
            # Check total annotation points
            count = group.get_total_node_count()
            print(f"Total annotation points: {count}")
            ```
        """
        return sum(tree.number_of_nodes() for tree in self.flattened_trees())

    def get_max_tree_id(self) -> int:
        """Returns the highest tree id of all trees within this group (and its subgroups)."""
        return max((tree.id for tree in self.flattened_trees()), default=0)

    def get_max_node_id(self) -> int:
        """Returns the highest node id of all nodes of all trees within this group (and its subgroups)."""
        return max(
            (tree.get_max_node_id() for tree in self.flattened_trees()),
            default=0,
        )

    def flattened_trees(self) -> Iterator[Tree]:
        """Returns an iterator of all trees in this group and its subgroups.

        This method performs a recursive traversal of the group hierarchy,
        yielding all trees regardless of their nesting level.

        Returns:
            Iterator[Tree]: Iterator yielding all contained trees.

        Examples:
            ```python
            # Process all trees regardless of grouping
            for tree in group.flattened_trees():
                print(f"Tree {tree.name} has {len(tree.nodes)} nodes")
            ```
        """
        yield from self.trees
        for group in self.groups:
            yield from group.flattened_trees()

    def flattened_groups(self) -> Iterator["Group"]:
        """Returns an iterator of all groups within this group (and its subgroups)."""
        for group in self.groups:
            yield group
            yield from group.flattened_groups()

    def get_node_by_id(self, node_id: int) -> "Node":
        """Retrieves a node by its ID from any tree in this group or its subgroups.

        Args:
            node_id: The ID of the node to find.

        Returns:
            Node: The node with the specified ID.

        Raises:
            ValueError: If no node with the given ID exists in any tree.

        Examples:
            ```python
            try:
                node = group.get_node_by_id(42)
                print(f"Found node at position {node.position}")
            except ValueError:
                print("Node not found")
            ```
        """
        for tree in self.flattened_trees():
            if tree.has_node(node_id):
                return tree.get_node_by_id(node_id)

        raise ValueError("Node id not found")

    def get_tree_by_id(self, tree_id: int) -> Tree:
        """Retrieves a tree by its ID from this group or its subgroups.

        Args:
            tree_id: The ID of the tree to find.

        Returns:
            Tree: The tree with the specified ID.

        Raises:
            ValueError: If no tree with the given ID exists.

        Examples:
            ```python
            try:
                tree = group.get_tree_by_id(42)
                print(f"Found tree '{tree.name}'")
            except ValueError:
                print("Tree not found")
            ```
        """
        # Todo: Use hashed access if it turns out to be worth it? # noqa: FIX002 Line contains TODO
        for tree in self.flattened_trees():
            if tree.id == tree_id:
                return tree
        raise ValueError(f"No tree with id {tree_id} was found")

    def has_tree_id(self, tree_id: int) -> bool:
        """Checks if a tree with the given ID exists in this group or its subgroups.

        Args:
            tree_id: The ID to check for.

        Returns:
            bool: True if a tree with the ID exists, False otherwise.

        Examples:
            ```python
            if group.has_tree_id(42):
                tree = group.get_tree_by_id(42)
                print(f"Tree exists: {tree.name}")
            ```
        """
        try:
            self.get_tree_by_id(tree_id)
            return True
        except ValueError:
            return False

    def get_group_by_id(self, group_id: int) -> "Group":
        """Returns the group which has the specified group id."""
        # Todo: Use hashed access if it turns out to be worth it? # noqa: FIX002 Line contains TODO
        for group in self.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No group with id {group_id} was found")

    def as_nml_group(self) -> wknml.Group:
        """Converts this group to its NML representation.

        This method creates a lightweight representation of the group
        suitable for serialization in the NML format.

        Returns:
            wknml.Group: NML representation of this group.

        Notes:
            This is primarily used internally for file I/O operations.
        """
        return wknml.Group(
            self.id,
            self.name,
            children=[g.as_nml_group() for g in self._child_groups],
        )

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and self._id == other._id

    def __hash__(self) -> int:
        return id(self)

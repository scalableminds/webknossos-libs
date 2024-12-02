import itertools
from os import PathLike
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import attr
import networkx as nx

from ..utils import warn_deprecated
from .group import Group

Vector3 = Tuple[float, float, float]


@attr.define
class Skeleton(Group):
    """A hierarchical representation of skeleton annotations in WEBKNOSSOS.

    The Skeleton class serves as the root container for all skeleton annotation data,
    organizing nodes and edges into a hierarchical structure of groups and trees.
    It contains dataset metadata and provides methods for loading, saving, and manipulating
    skeleton annotations.

    Attributes:
        voxel_size: 3D tuple (x, y, z) specifying the size of voxels in nanometers.
        dataset_name: Name of the dataset this skeleton belongs to.
        organization_id: Optional ID of the organization owning this skeleton.
        description: Optional description of the skeleton annotation.
        name: Always set to "Root" as this is the root group of the hierarchy.

    The skeleton structure follows a hierarchical organization:
        - Skeleton (root)
            - Groups (optional organizational units)
                - Trees (collections of connected nodes)
                    - Nodes (3D points with metadata)
                    - Edges (connections between nodes)

    Examples:
        Create and populate a new skeleton:
        ```python
        # Create skeleton through an annotation
        annotation = Annotation(
            name="dendrite_trace",
            dataset_name="cortex_sample",
            voxel_size=(11, 11, 24)
        )
        skeleton = annotation.skeleton

        # Add hierarchical structure
        dendrites = skeleton.add_group("dendrites")
        basal = dendrites.add_group("basal")
        tree = basal.add_tree("dendrite_1")

        # Add and connect nodes
        soma = tree.add_node(position=(100, 100, 100), comment="soma")
        branch = tree.add_node(position=(200, 150, 100), radius=1.5)
        tree.add_edge(soma, branch)
        ```

        Load an existing skeleton:
        ```python
        # Load from NML file
        skeleton = Skeleton.load("annotation.nml")

        # Access existing structure
        for group in skeleton.groups:
            for tree in group.trees:
                print(f"Tree {tree.name} has {len(tree.nodes)} nodes")
        ```

    Notes:
        - The Skeleton class inherits from Group, providing group and tree management methods.
        - To upload a skeleton to WEBKNOSSOS, create an Annotation with it.
        - For complex examples, see the skeleton synapse candidates example in the documentation.

    See Also:
        - Group: Base class providing group and tree management
        - Tree: Class representing connected node structures
        - Node: Class representing individual 3D points
        - Annotation: Container class for working with WEBKNOSSOS
    """

    voxel_size: Vector3
    dataset_name: str
    organization_id: Optional[str] = None
    description: Optional[str] = None
    # from Group parent to support mypy:
    _enforced_id: Optional[int] = attr.field(default=None, eq=False, repr=False)

    name: str = attr.field(default="Root", init=False, eq=False, repr=False)
    """Should not be used with `Skeleton`, this attribute is only useful for sub-groups. Set to `Root`."""

    # initialized in post_init:
    _id: int = attr.field(init=False, repr=False)
    _element_id_generator: Iterator[int] = attr.field(init=False, eq=False, repr=False)
    _skeleton: "Skeleton" = attr.field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._element_id_generator = itertools.count()
        self._skeleton = self
        super().__attrs_post_init__()  # sets self._id

    @property
    def scale(self) -> Tuple[float, float, float]:
        """Deprecated, please use `voxel_size`."""
        warn_deprecated("scale", "voxel_size")
        return self.voxel_size

    @scale.setter
    def scale(self, scale: Tuple[float, float, float]) -> None:
        """Deprecated, please use `voxel_size`."""
        warn_deprecated("scale", "voxel_size")
        self.voxel_size = scale

    @staticmethod
    def load(file_path: Union[PathLike, str]) -> "Skeleton":
        """Loads a `.nml` file or a `.zip` file containing an NML (and possibly also volume
        layers). Returns the `Skeleton` object. Also see `Annotation.load` if you want to
        have the annotation which wraps the skeleton."""

        from ..annotation import Annotation

        return Annotation.load(file_path).skeleton

    def save(self, out_path: Union[str, PathLike]) -> None:
        """
        Stores the skeleton as a zip or nml at the given path.
        """

        from ..annotation import Annotation

        out_path = Path(out_path)
        assert out_path.suffix in [
            ".nml",
            ".zip",
        ], f"The suffix if the file must be .nml or .zip, not {out_path.suffix}"
        annotation = Annotation(name=out_path.stem, skeleton=self, time=None)
        annotation.save(out_path)

    def add_nx_graphs(
        self, tree_dict: Union[List[nx.Graph], Dict[str, List[nx.Graph]]]
    ) -> None:
        """
        A utility to add nx graphs [NetworkX graph object](https://networkx.org/) to a wk skeleton object. Accepts both a simple list of multiple skeletons/trees or a dictionary grouping skeleton inputs.

        Arguments:
        tree_dict (Union[List[nx.Graph], Dict[str, List[nx.Graph]]]): A list of wK tree-like structures as NetworkX graphs or a dictionary of group names and same lists of NetworkX tree objects.
        """

        if not isinstance(tree_dict, dict):
            tree_dict = {"main_group": tree_dict}

        for group_name, trees in tree_dict.items():
            group = self.add_group(group_name)
            for tree in trees:
                tree_name = tree.graph.get("name", f"tree_{len(list(group.trees))}")
                wk_tree = group.add_tree(tree_name)
                wk_tree.color = tree.graph.get("color", None)
                id_node_dict = {}
                for id_with_node in tree.nodes(data=True):
                    old_id, node = id_with_node
                    node = wk_tree.add_node(
                        position=node.get("position"),
                        comment=node.get("comment", None),
                        radius=node.get("radius", 1.0),
                        rotation=node.get("rotation", None),
                        inVp=node.get("inVp", None),
                        inMag=node.get("inMag", None),
                        bitDepth=node.get("bitDepth", None),
                        interpolation=node.get("interpolation", None),
                        time=node.get("time", None),
                        is_branchpoint=node.get("is_branchpoint", False),
                        branchpoint_time=node.get("branchpoint_time", None),
                    )
                    id_node_dict[old_id] = node
                for edge in tree.edges():
                    wk_tree.add_edge(id_node_dict[edge[0]], id_node_dict[edge[1]])

    @staticmethod
    def from_path(file_path: Union[PathLike, str]) -> "Skeleton":
        """Deprecated. Use Skeleton.load instead."""
        warn_deprecated("Skeleton.from_path", "Skeleton.load")
        return Skeleton.load(file_path)

    def write(self, out_path: PathLike) -> None:
        """Deprecated. Use Skeleton.save instead."""
        warn_deprecated("Skeleton.write", "skeleton.save")
        self.save(out_path)

    def __hash__(self) -> int:
        return id(self)

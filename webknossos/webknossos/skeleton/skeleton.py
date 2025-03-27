import itertools
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import attr
import networkx as nx

from .group import Group

Vector3 = tuple[float, float, float]


@attr.define(eq=False)
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
    dataset_id: str | None = None
    organization_id: str | None = None
    description: str | None = None
    # from Group parent to support mypy:
    _enforced_id: int | None = attr.field(default=None, eq=False, repr=False)

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

    @staticmethod
    def load(file_path: PathLike | str) -> "Skeleton":
        """Load a skeleton annotation from a file.

        This method can load skeleton annotations from either a .nml file or a .zip file
        that contains an NML file. The .zip file may also contain volume layers.

        Args:
            file_path (PathLike | str): Path to the .nml or .zip file containing
                the skeleton annotation.

        Returns:
            Skeleton: A new Skeleton instance containing the loaded annotation data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file format is not supported or the file is corrupted.

        Examples:
            Load from an NML file:
            ```python
            # Load a simple NML file
            skeleton = Skeleton.load("dendrite_trace.nml")

            # Load from a ZIP archive containing NML and volume data
            skeleton = Skeleton.load("full_annotation.zip")
            ```

        Note:
            If you need access to volume layers or other annotation data, use
            `Annotation.load()` instead, which returns an Annotation object
            containing both the skeleton and any additional data.
        """
        from ..annotation import Annotation

        return Annotation.load(file_path).skeleton

    def save(self, out_path: str | PathLike) -> None:
        """Save the skeleton annotation to a file.

        Saves the skeleton data to either a .nml file or a .zip archive. The .zip
        format is recommended when the annotation includes volume layers or when
        you want to compress the data.

        Args:
            out_path (str | PathLike): Destination path for the saved file.
                Must end with either .nml or .zip extension.

        Raises:
            AssertionError: If the file extension is not .nml or .zip.
            OSError: If there are permission issues or insufficient disk space.

        Examples:
            ```python
            # Save as NML file
            skeleton.save("dendrite_annotation.nml")

            # Save as ZIP archive (recommended for complex annotations)
            skeleton.save("full_annotation.zip")
            ```

        Note:
            - The name of the annotation will be derived from the filename stem
            - When saving as .zip, any associated volume layers will be included
            - The .nml format is human-readable XML but may be larger in size
            - The .zip format provides compression and can store additional data
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
        self, tree_dict: list[nx.Graph] | dict[str, list[nx.Graph]]
    ) -> None:
        """Import NetworkX graphs as skeleton trees.

        Converts [NetworkX Graph objects](https://networkx.org) into skeleton trees, preserving node positions
        and edge connections. The graphs can be provided either as a list or as a
        dictionary mapping group names to lists of graphs.

        Args:
            tree_dict (list[nx.Graph] | dict[str, list[nx.Graph]]): Either:
                - A list of NetworkX graphs to be added directly to the skeleton
                - A dictionary mapping group names to lists of graphs, which will
                  create new groups with the specified names containing the graphs

        Raises:
            ValueError: If any graph nodes lack required position attributes.
            TypeError: If tree_dict is neither a list nor a dictionary.

        Examples:
            Add graphs directly to skeleton:
            ```python
            import networkx as nx

            # Create sample graphs
            g1 = nx.Graph()
            g1.add_node(1, position=(0, 0, 0))
            g1.add_node(2, position=(100, 0, 0))
            g1.add_edge(1, 2)

            g2 = nx.Graph()
            g2.add_node(1, position=(0, 100, 0))
            g2.add_node(2, position=(100, 100, 0))
            g2.add_edge(1, 2)

            # Add graphs directly
            skeleton.add_nx_graphs([g1, g2])

            # Or organize graphs into groups
            graphs_by_group = {
                "dendrites": [g1],
                "axons": [g2]
            }
            skeleton.add_nx_graphs(graphs_by_group)
            ```

        Note:
            - Each node in the input graphs must have a 'position' attribute
              containing (x, y, z) coordinates
            - Other node attributes (e.g., radius, rotation) will be preserved
            - Edge attributes are currently not preserved in the conversion
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

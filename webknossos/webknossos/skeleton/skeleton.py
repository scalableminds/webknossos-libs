import itertools
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import attr

from webknossos.utils import warn_deprecated

from .graph import Graph
from .group import Group

Vector3 = Tuple[float, float, float]

GroupOrGraph = Union[Group, Graph]


@attr.define()
class Skeleton(Group):
    """
    Representation of the [skeleton tracing](/webknossos/skeleton_annotation.html) of an `Annotation`.
    It contains metadata to identify the related dataset and is the root-group of sub-groups and graphs.
    See the parent class `Group` for methods about group and graph handling.
    To upload a skeleton to webknossos, please create an `Annotation()` with it.

    A small usage example:

    ```python
    annotation = Annotation("my_annotation", scale=(11, 11, 24))
    group = annotation.skeleton.add_group("a group")
    graph = group.add_graph("a graph")
    node_1 = graph.add_node(position=(0, 0, 0), comment="node 1")
    node_2 = graph.add_node(position=(100, 100, 100), comment="node 2")

    graph.add_edge(node_1, node_2)
    ```

    Also see [this example](/webknossos-py/examples/skeleton_synapse_candidates.html) for a more
    complex interaction.
    """

    scale: Vector3
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

    @classmethod
    def _set_init_docstring(cls) -> None:
        Skeleton.__init__.__doc__ = """
        To initialize a skeleton, setting the following parameters is required (or recommended):
        - scale
        - dataset_name
        - organization_id
        - description
        """

    def __attrs_post_init__(self) -> None:
        self._element_id_generator = itertools.count()
        self._skeleton = self
        super().__attrs_post_init__()  # sets self._id

    @staticmethod
    def load(file_path: Union[PathLike, str]) -> "Skeleton":
        """Loads a `.nml` file or a `.zip` file containing an NML (and possibly also volume
        layers). Returns the `Skeleton` object. Also see `Annotation.load` if you want to
        have the annotation which wraps the skeleton."""

        from webknossos import Annotation

        return Annotation.load(file_path).skeleton

    def save(self, out_path: Union[str, PathLike]) -> None:
        """
        Stores the skeleton as a zip or nml at the given path.
        """

        from webknossos import Annotation

        out_path = Path(out_path)
        assert (
            out_path.suffix == ".nml"
        ), f"The suffix if the file must be .nml, not {out_path.suffix}"
        annotation = Annotation(name=out_path.stem, skeleton=self, time=None)
        annotation.save(out_path)

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


Skeleton._set_init_docstring()

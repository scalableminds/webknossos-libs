import itertools
import warnings
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple, Union

import attr
from boltons.strutils import unit_len

from .graph import Graph
from .group import Group

Vector3 = Tuple[float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union[Group, Graph]


@attr.define()
class Skeleton(Group):
    """
    Representation of the [skeleton tracing](/webknossos/skeleton_annotation.html) of an `Annotation`.
    It contains metadata to identify the related dataset and is the root-group of sub-groups and graphs.
    See the parent class `Group` for methods about group and graph handling.
    To upload a skeleton to webknossos, please create an `Annotation()` with it.
    """

    scale: Vector3
    dataset_name: str
    organization_id: Optional[str] = None
    description: Optional[str] = None
    # from Group parent to support mypy:
    _enforced_id: Optional[int] = attr.field(default=None, eq=False, repr=False)
    name: str = attr.field(default="Root", init=False, eq=False, repr=False)
    """Should not be used with `Skeleton`, this attribute is only useful for sub-groups. Set to `Root`."""
    _children: Set[GroupOrGraph] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'children')}>",
    )
    # initialized in post_init:
    _id: int = attr.field(init=False, repr=False)
    _element_id_generator: Iterator[int] = attr.field(init=False, eq=False, repr=False)
    _skeleton: "Skeleton" = attr.field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._element_id_generator = itertools.count()
        self._skeleton = self
        super().__attrs_post_init__()  # sets self._id

    @staticmethod
    def load(file_path: Union[PathLike, str]) -> "Skeleton":
        from webknossos import Annotation

        return Annotation.load(file_path).skeleton

    def save(self, out_path: Union[str, PathLike]) -> None:
        from webknossos import Annotation

        out_path = Path(out_path)
        assert (
            out_path.suffix == ".nml"
        ), f"The suffix if the file must be .nml, not {out_path.suffix}"
        annotation = Annotation(name=out_path.stem, skeleton=self, time=None)
        annotation.save(out_path)

    @staticmethod
    def from_path(file_path: Union[PathLike, str]) -> "Skeleton":
        """Deprecated."""
        warnings.warn(
            "[DEPRECATION] Skeleton.from_path is deprecated, please use Skeleton.load instead."
        )
        return Skeleton.load(file_path)

    def write(self, out_path: PathLike) -> None:
        """Deprecated."""
        warnings.warn(
            "[DEPRECATION] skeleton.write is deprecated, please use skeleton.save instead."
        )
        self.save(out_path)

    def __hash__(self) -> int:
        return id(self)

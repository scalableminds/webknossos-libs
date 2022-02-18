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
    Contains metadata and is the root-group of sub-groups and graphs.
    See the parent class Group for methods about group and graph handling.
    """

    dataset_name: str
    scale: Vector3
    # from Group parent to support mypy:
    _enforced_id: Optional[int] = attr.field(default=None, eq=False, repr=False)
    name: str = attr.field(default="Root", init=False, eq=False, repr=False)
    _children: Set[GroupOrGraph] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'children')}>",
    )
    # initialized in post_init:
    _id: int = attr.field(init=False, repr=False)
    element_id_generator: Iterator[int] = attr.field(init=False, eq=False, repr=False)
    _skeleton: "Skeleton" = attr.field(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.element_id_generator = itertools.count()
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
        warnings.warn(
            "[DEPRECATION] Skeleton.from_path is deprecated, please use Skeleton.load instead."
        )
        return Skeleton.load(file_path)

    def write(self, out_path: PathLike) -> None:
        warnings.warn(
            "[DEPRECATION] skeleton.write is deprecated, please use skeleton.save instead."
        )
        self.save(out_path)

    def __hash__(self) -> int:
        return id(self)

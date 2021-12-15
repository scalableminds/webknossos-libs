import itertools
import warnings
from os import PathLike
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import attr
from boltons.strutils import unit_len

import webknossos.skeleton.nml as wknml
from webknossos._types import Openable
from webknossos.skeleton.nml.from_skeleton import from_skeleton as nml_from_skeleton
from webknossos.skeleton.nml.to_skeleton import to_skeleton as nml_to_skeleton
from webknossos.utils import time_since_epoch_in_ms

from .graph import Graph
from .group import Group

Vector3 = Tuple[float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union[Group, Graph]


@attr.define()
class Skeleton(Group):
    """
    Contains metadata and is the root-group of sub-groups and graphs.
    See the parent webknossos.skeleton.group.Group for methods about group and graph handling.
    """

    # from Group parent to support mypy:
    name: str
    _children: Set[GroupOrGraph] = attr.ib(
        factory=set,
        init=False,
        repr=lambda children: f"<{unit_len(children, 'children')}>",
    )

    scale: Vector3
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)
    offset: Optional[Vector3] = None
    time: Optional[int] = attr.ib(factory=time_since_epoch_in_ms)
    edit_position: Optional[Vector3] = None
    edit_rotation: Optional[Vector3] = None
    zoom_level: Optional[float] = None
    task_bounding_box: Optional[IntVector6] = None
    user_bounding_boxes: Optional[List[IntVector6]] = None

    _id: int = attr.ib(init=False, repr=False)
    element_id_generator: Iterator[int] = attr.ib(init=False, eq=False, repr=False)
    _skeleton: "Skeleton" = attr.ib(init=False, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.element_id_generator = itertools.count()
        self._skeleton = self
        super().__attrs_post_init__()

    @staticmethod
    def load(file_path: Union[Openable, PathLike, str]) -> "Skeleton":
        if isinstance(file_path, Openable):
            with file_path.open(mode="rb") as file_handle:
                return nml_to_skeleton(wknml.parse_nml(file_handle))
        else:
            with open(file_path, "rb") as file_handle:
                return nml_to_skeleton(wknml.parse_nml(file_handle))

    def save(self, out_path: Union[str, PathLike]) -> None:
        nml = nml_from_skeleton(
            self,
            self._get_nml_parameters(),
        )
        with open(out_path, "wb") as f:
            wknml.write_nml(f, nml)

    @staticmethod
    def from_path(file_path: Union[Openable, PathLike, str]) -> "Skeleton":
        warnings.warn(
            "[DEPRECATION] Skeleton.from_path is deprecated, please use Skeleton.load instead."
        )
        return Skeleton.load(file_path)

    def write(self, out_path: PathLike) -> None:
        warnings.warn(
            "[DEPRECATION] skeleton.write is deprecated, please use skeleton.save instead."
        )
        self.save(out_path)

    def _get_nml_parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scale": self.scale,
            "offset": self.offset,
            "time": self.time,
            "editPosition": self.edit_position,
            "editRotation": self.edit_rotation,
            "zoomLevel": self.zoom_level,
            "taskBoundingBox": self.task_bounding_box,
            "userBoundingBoxes": self.user_bounding_boxes,
        }

    def __hash__(self) -> int:
        return id(self)

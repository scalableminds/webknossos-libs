import itertools
from os import PathLike
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import attr

import webknossos.skeleton.nml as wknml
from webknossos.skeleton.nml.from_skeleton import from_skeleton as nml_from_skeleton
from webknossos.skeleton.nml.to_skeleton import to_skeleton as nml_to_skeleton

from .graph import Graph
from .group import Group
from .node import Node

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

GroupOrGraph = Union["Group", "Graph"]


@attr.define()
class Skeleton:
    """
    Contains groups and skeletons.
    """

    name: str
    scale: Vector3
    offset: Optional[Vector3] = None
    time: Optional[int] = None
    edit_position: Optional[Vector3] = None
    edit_rotation: Optional[Vector3] = None
    zoom_level: Optional[float] = None
    task_bounding_box: Optional[IntVector6] = None
    user_bounding_boxes: Optional[List[IntVector6]] = None

    root_group: Group = attr.ib(init=False)
    element_id_generator: Iterator[int] = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.element_id_generator = itertools.count()
        self.root_group = Group(name="Root", children=[], nml=self, is_root_group=False)
        self.time = int(str(self.time))  # typing: ignore

    def flattened_graphs(self) -> Generator["Graph", None, None]:
        return self.root_group.flattened_graphs()

    def get_graph_by_id(self, graph_id: int) -> Graph:
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for graph in self.root_group.flattened_graphs():
            if graph.id == graph_id:
                return graph
        raise ValueError(f"No graph with id {graph_id} was found")

    def get_group_by_id(self, group_id: int) -> Group:
        # Todo: Use hashed access if it turns out to be worth it? pylint: disable=fixme
        for group in self.root_group.flattened_groups():
            if group.id == group_id:
                return group
        raise ValueError(f"No group with id {group_id} was found")

    def add_graph(
        self,
        name: str,
        color: Optional[Vector4] = None,
        _nml: Optional["Skeleton"] = None,
        _enforced_id: Optional[int] = None,
    ) -> "Graph":
        return self.root_group.add_graph(
            name,
            color,
            _nml,
            _enforced_id,
        )

    def add_group(
        self, name: str, children: Optional[List[GroupOrGraph]] = None
    ) -> "Group":
        return self.root_group.add_group(name, children)

    def get_total_node_count(self) -> int:
        return self.root_group.get_total_node_count()

    def flattened_groups(self) -> Generator["Group", None, None]:
        return self.root_group.flattened_groups()

    def get_max_graph_id(self) -> int:
        return self.root_group.get_max_graph_id()

    def get_max_node_id(self) -> int:
        return self.root_group.get_max_node_id()

    def get_node_by_id(self, node_id: int) -> Node:
        return self.root_group.get_node_by_id(node_id)

    @staticmethod
    def from_path(file_path: PathLike) -> "Skeleton":
        with open(file_path, "rb") as file_handle:
            return nml_to_skeleton(wknml.parse_nml(file_handle))

    def write(self, out_path: PathLike) -> None:
        nml = nml_from_skeleton(
            self.root_group,
            self._get_nml_parameters(),
        )

        with open(out_path, "wb") as f:
            wknml.write_nml(f, nml)

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

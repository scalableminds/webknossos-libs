from os import PathLike

from webknossos.skeleton.graph import Graph
from webknossos.skeleton.group import Group
from webknossos.skeleton.node import Node
from webknossos.skeleton.skeleton import Skeleton


def open_nml(file_path: PathLike) -> Skeleton:
    return Skeleton.from_path(file_path)

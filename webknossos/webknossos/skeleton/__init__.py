import warnings
from os import PathLike
from typing import Union

from webknossos._types import Openable
from webknossos.skeleton.graph import Graph
from webknossos.skeleton.group import Group
from webknossos.skeleton.node import Node
from webknossos.skeleton.skeleton import Skeleton


def open_nml(file_path: Union[Openable, PathLike, str]) -> Skeleton:
    warnings.warn(
        "[DEPRECATION] open_nml is deprecated, please use Skeleton.load instead."
    )
    return Skeleton.load(file_path)

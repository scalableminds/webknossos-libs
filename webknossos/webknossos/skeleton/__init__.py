from os import PathLike
from typing import Union

from webknossos.skeleton.graph import Graph
from webknossos.skeleton.group import Group
from webknossos.skeleton.node import Node
from webknossos.skeleton.skeleton import Skeleton
from webknossos.utils import warn_deprecated


def open_nml(file_path: Union[PathLike, str]) -> Skeleton:
    """open_nml is deprecated, please use Skeleton.load instead."""
    warn_deprecated("open_nml", "Skeleton.load")
    return Skeleton.load(file_path)

# ruff: noqa: F401 imported but unused

from os import PathLike
from typing import Union

from ..utils import warn_deprecated
from .group import Group
from .node import Node
from .skeleton import Skeleton
from .tree import Graph, Tree


def open_nml(file_path: Union[PathLike, str]) -> Skeleton:
    """open_nml is deprecated, please use Skeleton.load instead."""
    warn_deprecated("open_nml", "Skeleton.load")
    return Skeleton.load(file_path)

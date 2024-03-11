from os import PathLike
from typing import Union

from ..utils import warn_deprecated
from .skeleton import Skeleton


def open_nml(file_path: Union[PathLike, str]) -> Skeleton:
    """open_nml is deprecated, please use Skeleton.load instead."""
    warn_deprecated("open_nml", "Skeleton.load")
    return Skeleton.load(file_path)

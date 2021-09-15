import attr
from typing import Tuple, List, Union
import numpy as np


@attr.frozen
class Vec3:
    x: int
    y: int
    z: int

    def __init__(self, vec3i_like: "Vec3Like"):
        self.x = vec3i_like.x
        self.y = vec3i_like.y
        self.z = vec3i_like.z


Vec3Like = Union[Vec3, Tuple[int, int, int], np.ndarray, List[int]]

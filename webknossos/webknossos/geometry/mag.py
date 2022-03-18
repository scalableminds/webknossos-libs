import math
import re
from functools import total_ordering
from math import log2
from typing import Any, Iterator, List, Optional, Tuple, cast

import attr
import numpy as np

from .vec3_int import Vec3Int, Vec3IntLike


def _import_mag(mag_like: Any) -> Vec3Int:
    as_vec3_int: Optional[Vec3Int] = None

    if isinstance(mag_like, Mag):
        as_vec3_int = mag_like.to_vec3_int()
    elif isinstance(mag_like, int):
        as_vec3_int = Vec3Int(mag_like, mag_like, mag_like)
    elif isinstance(mag_like, Vec3Int):
        as_vec3_int = mag_like
    elif isinstance(mag_like, list) or isinstance(mag_like, tuple):
        as_vec3_int = Vec3Int(cast(Vec3IntLike, mag_like))
    elif isinstance(mag_like, str):
        if re.match(r"^\d+$", mag_like) is not None:
            as_vec3_int = Vec3Int(int(mag_like), int(mag_like), int(mag_like))
        elif re.match(r"^\d+-\d+-\d+$", mag_like) is not None:
            as_vec3_int = Vec3Int([int(m) for m in mag_like.split("-")])
    elif isinstance(mag_like, np.ndarray):
        as_vec3_int = Vec3Int(mag_like)

    if as_vec3_int is None:
        raise ValueError(
            "Mag must be int or a vector3 of ints or a string shaped like e.g. 2-2-1"
        )
    for m in as_vec3_int:
        assert (
            log2(m) % 1 == 0
        ), f"Mag components must be power of 2, got {m} in {as_vec3_int}."

    return as_vec3_int


@total_ordering
@attr.frozen(order=False)
class Mag:
    """
    Represents the magnification level of a data layer. For example, the finest
    quality is usally not downsampled and is represented by Mag(1).
    When data is downsampled by a factor of 4 in all dimensions, this is referred
    to as Mag(4).
    When data is downsampled anisotropically by 2 in x and y and not downsampled in
    z, this is referred to as Mag(2, 2, 1).
    """

    _mag: Vec3Int = attr.ib(converter=_import_mag)

    @property
    def x(self) -> int:
        return self._mag.x

    @property
    def y(self) -> int:
        return self._mag.y

    @property
    def z(self) -> int:
        return self._mag.z

    @property
    def max_dim(self) -> int:
        return max(self._mag)

    @property
    def max_dim_log2(self) -> int:
        return int(math.log(self.max_dim) / math.log(2))

    def __lt__(self, other: Any) -> bool:
        return self.max_dim < Mag(other).max_dim

    def __le__(self, other: Any) -> bool:
        return self.max_dim <= Mag(other).max_dim

    def __eq__(self, other: Any) -> bool:
        return self.to_vec3_int() == Mag(other).to_vec3_int()

    def __str__(self) -> str:
        return self.to_layer_name()

    def __repr__(self) -> str:
        return f"Mag({self.to_layer_name()})"

    def to_layer_name(self) -> str:
        x, y, z = self._mag
        if x == y and y == z:
            return str(x)
        else:
            return self.to_long_layer_name()

    def to_long_layer_name(self) -> str:
        x, y, z = self._mag
        return "{}-{}-{}".format(x, y, z)

    def to_list(self) -> List[int]:
        return self._mag.to_list()

    def to_np(self) -> np.ndarray:
        return self._mag.to_np()

    def to_vec3_int(self) -> Vec3Int:
        return self._mag

    def to_tuple(self) -> Tuple[int, int, int]:
        return self._mag.to_tuple()

    def __mul__(self, factor: int) -> "Mag":
        return Mag(self._mag * factor)

    def __floordiv__(self, d: int) -> "Mag":
        return Mag(self._mag // d)

    def __hash__(self) -> int:
        return hash(self._mag)

    def __iter__(self) -> Iterator[int]:
        return iter(self._mag)

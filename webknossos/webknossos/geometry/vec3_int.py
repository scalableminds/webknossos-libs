from operator import add, floordiv, mod, mul, sub, lt, gt, eq
from typing import Any, Callable, List, Tuple, Union, Optional, Iterable

import attr
import numpy as np


@attr.frozen
class Vec3Int:
    x: int
    y: int
    z: int

    def __init__(
        self,
        vec: Union[int, "Vec3IntLike"],
        y: Optional[int] = None,
        z: Optional[int] = None,
    ):
        as_list: Optional[List[int]] = None
        value_error = "Vector components must be three integers or a Vec3IntLike object"

        if isinstance(vec, int):
            assert y is not None and z is not None, value_error
            as_list = [vec, y, z]
        else:
            assert y is None and z is None, value_error
            if isinstance(vec, list):
                as_list = vec
            elif isinstance(vec, tuple):
                as_list = [mag_d for mag_d in vec]
            elif isinstance(vec, np.ndarray):
                assert vec.shape == (
                    3,
                ), f"Numpy array for Vec3Int must have shape (3,), got {vec.shape}."
                as_list = list(vec)
            elif isinstance(vec, Vec3Int):
                as_list = vec.to_list()
        assert as_list is not None and len(as_list) == 3, value_error

        self.__attrs_init__(*as_list)

        # object.__setattr__(self, "x", as_list[0])
        # object.__setattr__(self, "y", as_list[1])
        # object.__setattr__(self, "z", as_list[2])

    def to_np(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.z]

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.x, self.y, self.z

    def _element_wise(
        self, other: Union[int, "Vec3IntLike"], fn: Callable[[int, Any], int]
    ) -> "Vec3Int":
        if isinstance(other, int):
            other_imported = Vec3Int(other, other, other)
        else:
            other_imported = Vec3Int(other)
        return Vec3Int(
            (
                fn(self.x, other_imported.x),
                fn(self.y, other_imported.y),
                fn(self.z, other_imported.z),
            )
        )

    def __add__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, add)

    def __sub__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, sub)

    def __mul__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, floordiv)

    def __mod__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, mod)

    def ceildiv(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return (self + other - 1) // other

    def pairmax(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, max)

    def pairmin(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, min)

    def __eq__(self, other: "Vec3IntLike") -> bool:
        return self.to_tuple() == other.to_tuple()

    def __repr__(self) -> str:
        return f"Vec3Int({self.x},{self.y},{self.z})"

    def __iter__(self) -> Iterable[int]:
        yield self.x
        yield self.y
        yield self.z

    @classmethod
    def zeros(cls) -> "Vec3Int":
        return cls((0, 0, 0))

    @classmethod
    def ones(cls) -> "Vec3Int":
        return cls((1, 1, 1))


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, List[int]]

from operator import add, floordiv, mod, mul, sub
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np


class Vec3Int(tuple):
    def __new__(
        cls,
        vec: Union[int, "Vec3IntLike"],
        y: Optional[int] = None,
        z: Optional[int] = None,
    ):
        if isinstance(vec, Vec3Int):
            return vec

        as_tuple: Optional[Tuple[int]] = None
        value_error = "Vector components must be three integers or a Vec3IntLike object"

        if isinstance(vec, int):
            assert y is not None and z is not None, value_error
            assert isinstance(y, int) and isinstance(z, int), value_error
            as_tuple = vec, y, z
        else:
            assert y is None and z is None, value_error
            if isinstance(vec, np.ndarray):
                assert vec.shape == (
                    3,
                ), f"Numpy array for Vec3Int must have shape (3,), got {vec.shape}."
            if isinstance(vec, Iterable):
                as_tuple = tuple(int(item) for item in vec)
        assert as_tuple is not None and len(as_tuple) == 3, value_error

        return super().__new__(cls, as_tuple)

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    @property
    def z(self) -> int:
        return self[2]

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

    def prod(self) -> int:
        self.x * self.y * self.z

    def __repr__(self) -> str:
        return f"Vec3Int({self.x},{self.y},{self.z})"

    @classmethod
    def zeros(cls) -> "Vec3Int":
        return cls(0, 0, 0)

    @classmethod
    def ones(cls) -> "Vec3Int":
        return cls(1, 1, 1)


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, List[int]]

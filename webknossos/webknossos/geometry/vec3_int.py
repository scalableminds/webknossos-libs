from operator import add, floordiv, mod, mul, sub
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

value_error = "Vector components must be three integers or a Vec3IntLike object."


class Vec3Int(tuple):
    def __new__(
        cls,
        vec: Union[int, "Vec3IntLike"],
        y: Optional[int] = None,
        z: Optional[int] = None,
    ) -> "Vec3Int":
        """
        Class to represent a 3D vector. Inherits from tuple and provides useful
        methods and operations on top.

        A small usage example:

        ```python
        from webknossos import Vec3Int

        vector_1 = Vec3Int(1, 2, 3)
        vector_2 = Vec3Int.full(1)
        assert vector_2.x == vector_2.y == vector_2.y

        assert vector_1 + vector_2 == (2, 3, 4)
        ```
        """

        if isinstance(vec, Vec3Int):
            return vec

        as_tuple: Optional[Tuple[int, int, int]] = None

        if isinstance(vec, int):
            assert y is not None and z is not None, value_error
            assert isinstance(y, int) and isinstance(z, int), value_error
            as_tuple = vec, y, z
        else:
            assert y is None and z is None, value_error
            if isinstance(vec, np.ndarray):
                assert np.count_nonzero(vec % 1) == 0, value_error
                assert vec.shape == (
                    3,
                ), "Numpy array for Vec3Int must have shape (3,)."
            if isinstance(vec, Iterable):
                as_tuple = cast(Tuple[int, int, int], tuple(int(item) for item in vec))
                assert len(as_tuple) == 3, value_error
        assert as_tuple is not None and len(as_tuple) == 3, value_error

        return super().__new__(cls, cast(Iterable, as_tuple))

    @staticmethod
    def from_xyz(x: int, y: int, z: int) -> "Vec3Int":
        """Use Vec3Int.from_xyz for fast construction."""

        # By calling __new__ of tuple directly, we circumvent
        # the tolerant (and potentially) slow Vec3Int.__new__ method.
        return tuple.__new__(Vec3Int, (x, y, z))

    @staticmethod
    def from_vec3_float(vec: Tuple[float, float, float]) -> "Vec3Int":

        return Vec3Int(int(vec[0]), int(vec[1]), int(vec[2]))

    @staticmethod
    def from_vec_or_int(vec_or_int: Union["Vec3IntLike", int]) -> "Vec3Int":
        if isinstance(vec_or_int, int):
            return Vec3Int.full(vec_or_int)
        else:
            return Vec3Int(vec_or_int)

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    @property
    def z(self) -> int:
        return self[2]

    def with_x(self, new_x: int) -> "Vec3Int":
        return Vec3Int.from_xyz(new_x, self.y, self.z)

    def with_y(self, new_y: int) -> "Vec3Int":
        return Vec3Int.from_xyz(self.x, new_y, self.z)

    def with_z(self, new_z: int) -> "Vec3Int":
        return Vec3Int.from_xyz(self.x, self.y, new_z)

    def to_np(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def to_list(self) -> List[int]:
        return [self.x, self.y, self.z]

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.x, self.y, self.z

    def contains(self, needle: int) -> bool:
        return self.x == needle or self.y == needle or self.z == needle

    def is_positive(self, strictly_positive: bool = False) -> bool:
        if strictly_positive:
            return all(i > 0 for i in self)
        else:
            return all(i >= 0 for i in self)

    def is_uniform(self) -> bool:
        return self.x == self.y == self.z

    def _element_wise(
        self, other: Union[int, "Vec3IntLike"], fn: Callable[[int, Any], int]
    ) -> "Vec3Int":
        if isinstance(other, int):
            other_imported = Vec3Int.from_xyz(other, other, other)
        else:
            other_imported = Vec3Int(other)
        return Vec3Int.from_xyz(
            fn(self.x, other_imported.x),
            fn(self.y, other_imported.y),
            fn(self.z, other_imported.z),
        )

    # note: (arguments incompatible with superclass, do not add Vec3Int to plain tuple! Hence the type:ignore)
    def __add__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":  # type: ignore[override]
        return self._element_wise(other, add)

    def __sub__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, sub)

    def __mul__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, floordiv)

    def __mod__(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, mod)

    def __neg__(self) -> "Vec3Int":
        return Vec3Int.from_xyz(-self.x, -self.y, -self.z)

    def ceildiv(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return (self + other - 1) // other

    def pairmax(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, max)

    def pairmin(self, other: Union[int, "Vec3IntLike"]) -> "Vec3Int":
        return self._element_wise(other, min)

    def prod(self) -> int:
        return self.x * self.y * self.z

    def __repr__(self) -> str:
        return f"Vec3Int({self.x},{self.y},{self.z})"

    def add_or_none(self, other: Optional["Vec3Int"]) -> Optional["Vec3Int"]:
        return None if other is None else self + other

    @classmethod
    def zeros(cls) -> "Vec3Int":
        return cls(0, 0, 0)

    @classmethod
    def ones(cls) -> "Vec3Int":
        return cls(1, 1, 1)

    @classmethod
    def full(cls, an_int: int) -> "Vec3Int":
        return cls(an_int, an_int, an_int)


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, Iterable[int]]

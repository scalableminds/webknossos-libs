import re
from typing import Iterable, Optional, Tuple, Type, Union, cast

import numpy as np

from .vec_int import VecInt

VALUE_ERROR = "Vector components must be three integers or a Vec3IntLike object."


class Vec3Int(VecInt):
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
            assert y is not None and z is not None, VALUE_ERROR
            assert isinstance(y, int) and isinstance(z, int), VALUE_ERROR
            as_tuple = vec, y, z
        else:
            assert y is None and z is None, VALUE_ERROR
            if isinstance(vec, np.ndarray):
                assert np.count_nonzero(vec % 1) == 0, VALUE_ERROR
                assert vec.shape == (
                    3,
                ), "Numpy array for Vec3Int must have shape (3,)."
            if isinstance(vec, Iterable):
                as_tuple = cast(Tuple[int, int, int], tuple(int(item) for item in vec))
        assert as_tuple is not None and len(as_tuple) == 3, VALUE_ERROR

        return cast(cls, super().__new__(cls, as_tuple))

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

        return Vec3Int(vec_or_int)

    @staticmethod
    def from_str(string: str) -> "Vec3Int":
        if re.match(r"\(\d+,\d+,\d+\)", string):
            return Vec3Int(tuple(map(int, re.findall(r"\d+", string))))

        return Vec3Int.full(int(string))

    @classmethod
    def zeros(cls, length: int = 3) -> "Vec3Int":
        del length

        return cls(0, 0, 0)

    @classmethod
    def ones(cls, length: int = 3) -> "Vec3Int":
        del length

        return cls(1, 1, 1)

    @classmethod
    def full(cls, an_int: int, length: int = 3) -> "Vec3Int":
        del length

        return cls(an_int, an_int, an_int)


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, Iterable[int]]

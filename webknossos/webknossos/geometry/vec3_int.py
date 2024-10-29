import re
from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np

from .vec_int import VecInt

_VALUE_ERROR = "Vector components must be three integers or a Vec3IntLike object."


class Vec3Int(VecInt):
    def __new__(
        cls,
        *args: Union["Vec3IntLike", Iterable[str], int],
        axes: Optional[Iterable[str]] = ("x", "y", "z"),
        **kwargs: int,
    ) -> "Vec3Int":
        """A 3D vector class that inherits from tuple with additional vector operations.

        This class provides a convenient way to work with 3D integer vectors, supporting
        common vector operations and component access.

        Examples:
            Basic vector creation and operations:
            ```
            vector_1 = Vec3Int(1, 2, 3)
            vector_2 = Vec3Int.full(1)
            assert vector_2.x == vector_2.y == vector_2.y
            assert vector_1 + vector_2 == (2, 3, 4)
            ```
        """

        if args:
            if isinstance(args[0], Vec3Int):
                return args[0]

            assert axes is not None, _VALUE_ERROR

            if isinstance(args[0], Iterable):
                self = super().__new__(cls, *args[0], axes=("x", "y", "z"))
                assert self is not None and len(self) == 3, _VALUE_ERROR

                return cast(Vec3Int, self)

            assert len(args) == 3 and len(tuple(axes)) == 3, _VALUE_ERROR
            assert kwargs is None or len(kwargs) == 0, _VALUE_ERROR
            assert "x" in axes and "y" in axes and "z" in axes, _VALUE_ERROR
            values, _ = zip(*sorted(zip(args, axes), key=lambda x: x[1]))
        else:
            assert "x" in kwargs and "y" in kwargs and "z" in kwargs, _VALUE_ERROR
            assert len(kwargs) == 3, _VALUE_ERROR
            values = kwargs["x"], kwargs["y"], kwargs["z"]

        self = super().__new__(cls, *values, axes=("x", "y", "z"))
        self.axes = ("x", "y", "z")

        assert self is not None and len(self) == 3, _VALUE_ERROR

        return cast(Vec3Int, self)

    def __getnewargs__(self) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        return (self.to_tuple(), self.axes)

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

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @staticmethod
    def from_xyz(x: int, y: int, z: int) -> "Vec3Int":
        """Use Vec3Int.from_xyz for fast construction."""

        # By calling __new__ of tuple directly, we circumvent
        # the tolerant (and potentially) slow Vec3Int.__new__ method.
        vec3int = tuple.__new__(Vec3Int, (x, y, z))
        vec3int.axes = ("x", "y", "z")
        return vec3int

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
        if re.match(r"^\(\d+,\d+,\d+\)$", string):
            # matches a string that consists of three comma-separated digits in parentheses
            return Vec3Int(tuple(map(int, re.findall(r"\d+", string))))
        elif re.match(r"^\d+,\d+,\d+$", string):
            # matches a string that consists of three digits separated by commas
            return Vec3Int(tuple(map(int, string.split(","))))

        return Vec3Int.full(int(string))

    @classmethod
    def zeros(cls, _axes: Tuple[str, ...] = ("x", "y", "z")) -> "Vec3Int":
        return cls(0, 0, 0)

    @classmethod
    def ones(cls, _axes: Tuple[str, ...] = ("x", "y", "z")) -> "Vec3Int":
        return cls(1, 1, 1)

    @classmethod
    def full(cls, an_int: int, _axes: Tuple[str, ...] = ("x", "y", "z")) -> "Vec3Int":
        return cls(an_int, an_int, an_int)


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, Iterable[int]]

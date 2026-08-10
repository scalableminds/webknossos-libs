import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from operator import add, mul, sub, truediv
from typing import Any, TypeAlias, Union

import numpy as np

from .vec3_int import Vec3Int

_VALUE_ERROR = "Vector components must be three floats, got {}."


class Vec3Float(Sequence[float]):
    """A three-dimensional vector of floats, e.g. a voxel size or a position with
    sub-voxel precision. Use `Vec3Int` where only whole voxels are meaningful.

    Examples:
        Basic vector creation and operations:
        ```
        vector_1 = Vec3Float(1.0, 2.0, 3.0)
        vector_2 = Vec3Float.full(1.0)
        assert vector_2.x == vector_2.y == vector_2.z
        assert vector_1 + vector_2 == (2.0, 3.0, 4.0)
        ```

    Instances compare, order and hash like the equivalent plain tuple, so they can be
    used interchangeably with `(x, y, z)` in comparisons, sorting, sets and as dict
    keys.
    """

    # Instances are created in large numbers, e.g. one per node of a skeleton, so they
    # do without the per-instance `__dict__` that a plain class would carry.
    __slots__ = ("_data",)

    _data: tuple[float, float, float]

    def __new__(cls, *args: Union["Vec3FloatLike", float]) -> "Vec3Float":
        if len(args) == 3:
            # Fast path for the three-scalar case, which avoids numpy. This matters
            # because the constructor is used as an attrs converter.
            first, second, third = args
            if (
                isinstance(first, (int, float))
                and isinstance(second, (int, float))
                and isinstance(third, (int, float))
            ):
                return cls.from_xyz(float(first), float(second), float(third))
            raise ValueError(_VALUE_ERROR.format(args))

        if len(args) != 1:
            raise ValueError(_VALUE_ERROR.format(args))

        value = args[0]
        if isinstance(value, Vec3Float):
            return value

        # Fast path for a single tuple of numbers, which is the shape the attrs
        # converters actually get, e.g. for every node of a skeleton read from an NML
        if type(value) is tuple and len(value) == 3:
            first, second, third = value
            if (
                isinstance(first, (int, float))
                and isinstance(second, (int, float))
                and isinstance(third, (int, float))
            ):
                return cls.from_xyz(float(first), float(second), float(third))

        if not isinstance(value, np.ndarray):
            if not isinstance(value, Iterable):
                raise ValueError(_VALUE_ERROR.format(repr(value)))
            value = tuple(value)
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(_VALUE_ERROR.format(repr(value))) from None
        if array.shape != (3,):
            raise ValueError(_VALUE_ERROR.format(f"an object of shape {array.shape}"))
        return cls.from_xyz(float(array[0]), float(array[1]), float(array[2]))

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> "Vec3Float":
        """Creates a `Vec3Float` from three floats, skipping the general parsing."""
        # `_data` is set in `__new__` rather than `__init__` so that pickling and
        # deepcopying work: a half-initialized object would otherwise raise on
        # attribute access.
        self = object.__new__(cls)
        self._data = (x, y, z)
        return self

    @classmethod
    def from_str(cls, string: str) -> "Vec3Float":
        """Parses a `Vec3Float` from a string such as `"1.0,2.0,3.0"` or `"(1.0,2.0,3.0)"`."""
        return cls(
            tuple(float(part) for part in re.split(r"[,\s]+", string.strip("()")))
        )

    @classmethod
    def zeros(cls) -> "Vec3Float":
        return cls.from_xyz(0.0, 0.0, 0.0)

    @classmethod
    def ones(cls) -> "Vec3Float":
        return cls.from_xyz(1.0, 1.0, 1.0)

    @classmethod
    def full(cls, value: float) -> "Vec3Float":
        """Creates a `Vec3Float` with all three components set to `value`."""
        return cls.from_xyz(float(value), float(value), float(value))

    @property
    def x(self) -> float:
        return self._data[0]

    @property
    def y(self) -> float:
        return self._data[1]

    @property
    def z(self) -> float:
        return self._data[2]

    def with_x(self, new_x: float) -> "Vec3Float":
        return Vec3Float.from_xyz(float(new_x), self.y, self.z)

    def with_y(self, new_y: float) -> "Vec3Float":
        return Vec3Float.from_xyz(self.x, float(new_y), self.z)

    def with_z(self, new_z: float) -> "Vec3Float":
        return Vec3Float.from_xyz(self.x, self.y, float(new_z))

    def __getitem__(self, index: Any) -> Any:
        return self._data[index]

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __hash__(self) -> int:
        return hash(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vec3Float):
            return self._data == other._data
        if isinstance(other, tuple):
            return self._data == other
        return NotImplemented

    def _comparable_data(self, other: object) -> tuple[float, ...] | None:
        """Returns the data of `other` to compare against, or None if it is not comparable."""
        if isinstance(other, Vec3Float):
            return other._data
        if isinstance(other, tuple):
            return other
        return None

    # The vectors are ordered lexicographically, just like the equivalent plain tuples.
    def __lt__(self, other: object) -> bool:
        other_data = self._comparable_data(other)
        if other_data is None:
            return NotImplemented
        return self._data < other_data

    def __le__(self, other: object) -> bool:
        other_data = self._comparable_data(other)
        if other_data is None:
            return NotImplemented
        return self._data <= other_data

    def __gt__(self, other: object) -> bool:
        other_data = self._comparable_data(other)
        if other_data is None:
            return NotImplemented
        return self._data > other_data

    def __ge__(self, other: object) -> bool:
        other_data = self._comparable_data(other)
        if other_data is None:
            return NotImplemented
        return self._data >= other_data

    def __getnewargs__(self) -> tuple[tuple[float, float, float]]:
        return (self._data,)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = np.array(self._data, dtype=dtype or np.float64)
        return array

    def to_tuple(self) -> tuple[float, float, float]:
        return self._data

    def to_list(self) -> list[float]:
        return list(self._data)

    def to_np(self) -> np.ndarray:
        return np.array(self._data, dtype=np.float64)

    def to_vec3_int(self) -> Vec3Int:
        """Converts to a `Vec3Int`, truncating each component towards zero."""
        return Vec3Int.from_xyz(int(self.x), int(self.y), int(self.z))

    def _element_wise(
        self, other: Union[float, "Vec3FloatLike"], fn: Callable[[float, float], float]
    ) -> "Vec3Float":
        if isinstance(other, (int, float)):
            other_imported = Vec3Float.full(other)
        else:
            other_imported = Vec3Float(other)
        return Vec3Float.from_xyz(
            fn(self._data[0], other_imported._data[0]),
            fn(self._data[1], other_imported._data[1]),
            fn(self._data[2], other_imported._data[2]),
        )

    # Note: Adding plain tuples concatenates them, whereas Vec3Float adds the elements
    # at the same index, matching Vec3Int.
    def __add__(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        return self._element_wise(other, add)

    def __sub__(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        return self._element_wise(other, sub)

    # Note: Multiplying a plain tuple by an int repeats it, which again differs.
    def __mul__(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        return self._element_wise(other, mul)

    def __truediv__(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        return self._element_wise(other, truediv)

    def __neg__(self) -> "Vec3Float":
        return Vec3Float.from_xyz(-self.x, -self.y, -self.z)

    def pairmax(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        """Returns the element-wise maximum of the two vectors."""
        return self._element_wise(other, max)

    def pairmin(self, other: Union[float, "Vec3FloatLike"]) -> "Vec3Float":
        """Returns the element-wise minimum of the two vectors."""
        return self._element_wise(other, min)

    def prod(self) -> float:
        """Returns the product of the three components."""
        return self.x * self.y * self.z

    def __repr__(self) -> str:
        return f"Vec3Float({self.x},{self.y},{self.z})"


Vec3FloatLike: TypeAlias = (
    Vec3Float | Vec3Int | tuple[float, float, float] | np.ndarray | Iterable[float]
)
"""Anything the `Vec3Float` constructor can turn into a `Vec3Float`."""


def as_vec3_float_or_none(
    vec3_float_like: Vec3FloatLike | None,
) -> Vec3Float | None:
    """Like the `Vec3Float` constructor, but passes `None` through.

    Not part of the public API, import it from this module if you need it.

    Exists as a module-level function rather than a classmethod because mypy's attrs
    plugin only accepts named functions, types and lambdas as field converters.
    """
    if vec3_float_like is None:
        return None
    return Vec3Float(vec3_float_like)

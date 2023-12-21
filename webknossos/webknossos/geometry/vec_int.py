import re
from operator import add, floordiv, mod, mul, sub
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

VALUE_ERROR = "VecInt can be instantiated with int values `VecInt(1,2,3,4) or with `VecIntLike` object `VecInt([1,2,3,4])."


class VecInt(tuple):
    def __new__(
        cls,
        vec: Union[int, "VecIntLike"],
        *args: int,
        y: Optional[int] = None,
        z: Optional[int] = None,
        **kwargs: int,
    ) -> "VecInt":
        """
        Class to represent a ND vector. Inherits from tuple and provides useful
        methods and operations on top. The vector has a minimal length of 3 and
        it is assumed that the first three axes are `x`, `y` and `z`.

        A small usage example:

        ```python
        from webknossos import VecInt

        vector_1 = VecInt(1, y=2, z=3, t=4)
        vector_2 = VecInt.full(1, 4)
        assert vector_2[0] == vector_2[1] == vector_2[2] == vector_2[3]

        assert vector_1 + vector_2 == (2, 3, 4, 5)
        ```
        """

        if isinstance(vec, VecInt):
            return vec

        as_tuple: Optional[Tuple[int, ...]] = None

        if isinstance(vec, int):
            if args:
                assert all(isinstance(arg, int) for arg in args), VALUE_ERROR
                as_tuple = (vec, *args)
            else:
                assert y is not None and z is not None, VALUE_ERROR
                remaining_axes = []
                for key, value in kwargs.items():
                    assert key not in ["x", "y", "z"]
                    remaining_axes.append((key, value))
                as_tuple = (
                    vec,
                    y,
                    z,
                    *[x[1] for x in sorted(remaining_axes, key=lambda x: x[0])],
                )
        else:
            if args:
                raise ValueError(VALUE_ERROR)
            if isinstance(vec, np.ndarray):
                assert np.count_nonzero(vec % 1) == 0, VALUE_ERROR
            if isinstance(vec, str):
                return cls.from_str(vec)
            if isinstance(vec, Iterable):
                as_tuple = cast(Tuple[int, ...], tuple(int(item) for item in vec))
        assert as_tuple is not None and len(as_tuple) >= 3, VALUE_ERROR

        return super().__new__(cls, cast(Iterable, as_tuple))

    @staticmethod
    def from_str(string: str) -> "VecInt":
        return VecInt(tuple(map(int, re.findall(r"\d+", string))))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    @property
    def z(self) -> int:
        return self[2]

    def with_x(self, new_x: int) -> "VecInt":
        return self.__class__(new_x, self.y, self.z, *self[3:])

    def with_y(self, new_y: int) -> "VecInt":
        return self.__class__(self.x, new_y, self.z, *self[3:])

    def with_z(self, new_z: int) -> "VecInt":
        return self.__class__(self.x, self.y, new_z, *self[3:])

    def with_replaced(self, index: int, new_element: int) -> "VecInt":
        """Returns a new ND Vector with a replaced element at a given index."""

        return VecInt(*self[:index], new_element, *self[index + 1 :])

    def to_np(self) -> np.ndarray:
        return np.array(self)

    def to_list(self) -> List[int]:
        return list(self)

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self)

    def to_xyz(self) -> Tuple[int, int, int]:
        """Returns the x, y and z component of the n dimensional vector. Other axes are ignored."""
        return (self.x, self.y, self.z)

    def contains(self, needle: int) -> bool:
        return any(element == needle for element in self)

    def is_positive(self, strictly_positive: bool = False) -> bool:
        if strictly_positive:
            return all(i > 0 for i in self)

        return all(i >= 0 for i in self)

    def is_uniform(self) -> bool:
        first = self[0]
        return all(element == first for element in self)

    def _element_wise(
        self, other: Union[int, "VecIntLike"], fn: Callable[[int, Any], int]
    ) -> "VecInt":
        if isinstance(other, int):
            other_imported = VecInt.full(other, len(self))
        else:
            other_imported = VecInt(other)
            assert len(other_imported) == len(
                self
            ), f"{other} and {self} are not equally shaped."
        return self.__class__(
            (fn(self[i], other_imported[i]) for i in range(len(self)))
        )

    # note: (arguments incompatible with superclass, do not add VecInt to plain tuple! Hence the type:ignore)
    def __add__(self, other: Union[int, "VecIntLike"]) -> "VecInt":  # type: ignore[override]
        return self._element_wise(other, add)

    def __sub__(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return self._element_wise(other, sub)

    # Note: When multiplying regular tuples with an int those are repeated,
    # which is a different behavior in the superclass! Hence the type:ignore.
    def __mul__(self, other: Union[int, "VecIntLike"]) -> "VecInt":  # type: ignore[override]
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return self._element_wise(other, floordiv)

    def __mod__(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return self._element_wise(other, mod)

    def __neg__(self) -> "VecInt":
        return self.__class__(-elem for elem in self)

    def ceildiv(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return (self + other - 1) // other

    def pairmax(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return self._element_wise(other, max)

    def pairmin(self, other: Union[int, "VecIntLike"]) -> "VecInt":
        return self._element_wise(other, min)

    def prod(self) -> int:
        return int(np.prod(self))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join((str(element) for element in self))})"

    def add_or_none(self, other: Optional["VecInt"]) -> Optional["VecInt"]:
        return None if other is None else self + other

    def moveaxis(
        self, source: Union[int, List[int]], target: Union[int, List[int]]
    ) -> "VecInt":
        """
        Allows to move one element at index `source` to another index `target`. Similar to
        np.moveaxis, this is *not* a swap operation but instead it moves the specified
        source so that the other elements move when necessary.
        """

        # Piggy-back on np.moveaxis by creating an auxiliary array where the indices 0, 1 and
        # 2 appear in the shape.
        indices = np.moveaxis(
            np.zeros(tuple(i for i in range(len(self)))), source, target
        ).shape
        arr = self.to_np()[np.array(indices)]
        return self.__class__(arr)

    @classmethod
    def zeros(cls, length: int) -> "VecInt":
        return cls((0 for _ in range(length)))

    @classmethod
    def ones(cls, length: int) -> "VecInt":
        return cls((1 for _ in range(length)))

    @classmethod
    def full(cls, an_int: int, length: int) -> "VecInt":
        return cls((an_int for _ in range(length)))


VecIntLike = Union[VecInt, Tuple[int, ...], np.ndarray, Iterable[int]]

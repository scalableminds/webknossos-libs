import re
from operator import add, floordiv, mod, mul, sub
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

VALUE_ERROR = "VecNInt can be instantiated with int values `VecNInt(1,2,3,4) or with `VecNIntLike` object `VecNInt([1,2,3,4])."


class VecNInt(tuple):
    def __new__(
        cls,
        vec: Union[int, "VecNIntLike"],
        *args: int,
    ) -> "VecNInt":
        """
        Class to represent a ND vector. Inherits from tuple and provides useful
        methods and operations on top. The vector has a minimal length of 3 and
        it is assumed that the first three axes are `x`, `y` and `z`.

        A small usage example:

        ```python
        from webknossos import VecNInt

        vector_1 = VecNInt(1, 2, 3, 4)
        vector_2 = VecNInt.full(1, 4)
        assert vector_2[0] == vector_2[1] == vector_2[2] == vector_2[3]

        assert vector_1 + vector_2 == (2, 3, 4, 5)
        ```
        """

        if isinstance(vec, VecNInt):
            return vec

        as_tuple: Optional[Tuple[int, ...]] = None

        if isinstance(vec, int):
            assert all(isinstance(arg, int) for arg in args), VALUE_ERROR
            as_tuple = (vec, *args)
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
    def from_str(string: str) -> "VecNInt":
        return VecNInt(tuple(map(int, re.findall(r"\d+", string))))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    @property
    def z(self) -> int:
        return self[2]

    def with_x(self, new_x: int) -> "VecNInt":
        return self.__class__(new_x, self.y, self.z, *self[3:])

    def with_y(self, new_y: int) -> "VecNInt":
        return self.__class__(self.x, new_y, self.z, *self[3:])

    def with_z(self, new_z: int) -> "VecNInt":
        return self.__class__(self.x, self.y, new_z, *self[3:])

    def with_replaced(self, index: int, new_element: int) -> "VecNInt":
        """Returns a new ND Vector with a replaced element at a given index."""

        return VecNInt(*self[:index], new_element, *self[index + 1:])

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
        self, other: Union[int, "VecNIntLike"], fn: Callable[[int, Any], int]
    ) -> "VecNInt":
        if isinstance(other, int):
            other_imported = VecNInt.full(other, len(self))
        else:
            other_imported = VecNInt(other)
            assert len(other_imported) == len(self), f"{other} and {self} are not equally shaped."
        return self.__class__(
            (fn(other_imported[i], self[i]) for i in range(len(self)))
        )

    # note: (arguments incompatible with superclass, do not add VecNInt to plain tuple! Hence the type:ignore)
    def __add__(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":  # type: ignore[override]
        return self._element_wise(other, add)

    def __sub__(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return self._element_wise(other, sub)

    # Note: When multiplying regular tuples with an int those are repeated,
    # which is a different behavior in the superclass! Hence the type:ignore.
    def __mul__(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":  # type: ignore[override]
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return self._element_wise(other, floordiv)

    def __mod__(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return self._element_wise(other, mod)

    def __neg__(self) -> "VecNInt":
        return self.__class__(-elem for elem in self)

    def ceildiv(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return (self + other - 1) // other

    def pairmax(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return self._element_wise(other, max)

    def pairmin(self, other: Union[int, "VecNIntLike"]) -> "VecNInt":
        return self._element_wise(other, min)

    def prod(self) -> int:
        return np.prod(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join((str(element) for element in self))})"

    def add_or_none(self, other: Optional["VecNInt"]) -> Optional["VecNInt"]:
        return None if other is None else self + other

    def moveaxis(
        self, source: Union[int, List[int]], target: Union[int, List[int]]
    ) -> "VecNInt":
        """
        Allows to move one element at index `source` to another index `target`. Similar to
        np.moveaxis, this is *not* a swap operation but instead it moves the specified
        source so that the other elements move when necessary.
        """

        # Piggy-back on np.moveaxis by creating an auxiliary array where the indices 0, 1 and
        # 2 appear in the shape.
        indices = np.moveaxis(np.zeros(tuple(i for i in range(len(self)))), source, target).shape
        arr = self.to_np()[np.array(indices)]
        return self.__class__(arr)

    @classmethod
    def zeros(cls, length: int) -> "VecNInt":
        return cls((0 for _ in range(length)))

    @classmethod
    def ones(cls, length: int) -> "VecNInt":
        return cls((1 for _ in range(length)))

    @classmethod
    def full(cls, an_int: int, length: int) -> "VecNInt":
        return cls((an_int for _ in range(length)))


VecNIntLike = Union[VecNInt, Tuple[int, ...], np.ndarray, Iterable[int]]

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from operator import add, floordiv, mod, mul, sub
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

import numpy as np

if TYPE_CHECKING:
    from .vec3_int import Vec3Int


def _value_error(args: Any) -> str:
    return (
        'VecInt can be instantiated with int values `VecInt(1, 2, 3, 4, axes=("x","y","z","t"))`, '
        + "with axes as argument names `VecInt(x=10, y=42, z=3, t=100)` or with two Iterables one with the "
        + 'axes values and the other with axes names `VecInt((1, 2, 3, 4), ("x", "y", "z", "t"))`. '
        + f"Got {args}."
    )


_T = TypeVar("_T", bound="VecInt")


class VecInt(Sequence[int]):
    """
    A specialized vector class for storing and manipulating integer values with named axes.

    This class uses composition (an internal `_data` tuple) to provide vector operations
    while preserving axis information. It allows for initialization with both positional
    and named arguments.

    Attributes:
        axes (tuple[str, ...]): Names of the vector's axes, e.g. ('x', 'y', 'z')

    Examples:
        Create a vector with 4 named dimensions:
        ```
        vector_1 = VecInt(1, 2, 3, 4, axes=("x", "y", "z", "t"))
        vector_1 = VecInt([1, 2, 3, 4], axes=("x", "y", "z", "t"))
        vector_1 = VecInt(x=1, y=2, z=3, t=4)
        ```

        Create a vector filled with ones:
        ```
        vector_2 = VecInt.full(1, axes=("x", "y", "z", "t"))
        assert vector_2[0] == vector_2[1] == vector_2[2] == vector_2[3]
        ```

        Perform vector addition:
        ```
        assert vector_1 + vector_2 == VecInt(2, 3, 4, 5)
        ```
    """

    _data: tuple[int, ...]
    axes: tuple[str, ...]
    _c_pos: int | None
    _x_pos: int | None
    _y_pos: int | None
    _z_pos: int | None

    def __new__(
        cls,
        *args: Union["VecIntLike", Iterable[str], int],  # noqa: UP007
        axes: Iterable[str] | None = None,
        **kwargs: int,
    ) -> "VecInt":
        as_tuple: tuple[int, ...] | None = None

        if args:
            if isinstance(args[0], VecInt):
                if axes is not None:
                    axes = tuple(axes)
                    if len(axes) != len(args[0]):
                        raise ValueError(
                            "The number of axes must match the number of axes in the VecInt."
                        )
                    return cls(args[0].to_tuple(), axes=axes)
                return args[0]
            if isinstance(args[0], np.ndarray):
                assert np.count_nonzero(args[0] % 1) == 0, _value_error(args)
            if isinstance(args[0], str):
                return cls.from_str(args[0])
            if isinstance(args[0], Iterable):
                as_tuple = tuple(int(item) for item in args[0])
                if args[1:] and isinstance(args[1], Iterable):
                    assert all(isinstance(arg, str) for arg in args[1]), _value_error(
                        args
                    )
                    axes = tuple(args[1])  # type: ignore
            elif isinstance(args, Iterable):
                as_tuple = tuple(int(arg) for arg in args)  # type: ignore
            else:
                raise ValueError(_value_error(args))
            assert axes is not None, _value_error(args)
        else:
            assert kwargs, _value_error(args)
            assert axes is None, _value_error(args)
            as_tuple = tuple(kwargs.values())

        assert as_tuple is not None, _value_error(args)

        self = object.__new__(cls)
        self._data = as_tuple
        # self.axes is set in __new__ instead of __init__ so that pickling/unpickling
        # works without problems. As long as the deserialization of a tree instance
        # is not finished, the object is only half-initialized. Since self.axes
        # is needed after deepcopy, an error would be raised otherwise.
        # Also see:
        # https://stackoverflow.com/questions/46283738/attributeerror-when-using-python-deepcopy
        self.axes = tuple(axes or kwargs.keys())
        self._c_pos = self.axes.index("c") if "c" in self.axes else None
        self._x_pos = self.axes.index("x") if "x" in self.axes else None
        self._y_pos = self.axes.index("y") if "y" in self.axes else None
        self._z_pos = self.axes.index("z") if "z" in self.axes else None

        return self

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            if index not in self.axes:
                raise KeyError(f"Axis {index} not found in {self.axes}")
            return self[self.axes.index(index)]
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __hash__(self) -> int:
        return hash(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VecInt):
            return self._data == other._data
        if isinstance(other, tuple):
            return self._data == other
        return NotImplemented

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def __getnewargs__(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        return (self.to_tuple(), self.axes)

    @property
    def c(self) -> int:
        """
        Returns the c component of the vector.
        """
        if self._c_pos is not None:
            return self[self._c_pos]

        raise KeyError("The vector does not have an c component.")

    @property
    def x(self) -> int:
        """
        Returns the x component of the vector.
        """
        if self._x_pos is not None:
            return self[self._x_pos]

        raise KeyError("The vector does not have an x component.")

    @property
    def y(self) -> int:
        """
        Returns the y component of the vector.
        """
        if self._y_pos is not None:
            return self[self._y_pos]

        raise KeyError("The vector does not have an y component.")

    @property
    def z(self) -> int:
        """
        Returns the z component of the vector.
        """
        if self._z_pos is not None:
            return self[self._z_pos]

        raise KeyError("The vector does not have an z component.")

    @property
    def xyz(self) -> "Vec3Int":
        from .vec3_int import Vec3Int

        return Vec3Int(self.x, self.y, self.z)

    def get(self, axis: str, default: int | None = None) -> int:
        """
        Returns the value of the specified axis.

        Args:
            axis (str): The name of the axis to get the value for.
            default (int, optional): The default value to return if the axis is not present. Defaults to None.

        Returns:
            int: The value of the specified axis.
        """
        if axis in self.axes:
            return self[self.axes.index(axis)]
        else:
            if default is None:
                raise KeyError(f"Axis {axis} not found in {self.axes}")
            return default

    @staticmethod
    def from_str(string: str) -> "VecInt":
        """
        Returns a new ND Vector from a string representation.

        Args:
            string (str): The string representation of the vector.

        Returns:
            VecInt: The new vector.
        """
        return VecInt(tuple(map(int, re.findall(r"\d+", string))))

    def with_replaced(self: _T, axis: str | int, new_element: int) -> _T:
        """Returns a new ND Vector with a replaced element at a given axis (or index for backwards compatibility)."""

        if isinstance(axis, int):
            index = axis
        else:
            index = self.axes.index(axis)

        return self.__class__(
            *self._data[:index], new_element, *self._data[index + 1 :], axes=self.axes
        )

    def with_c(self: _T, new_c: int) -> _T:
        """Returns a new ND Vector with the c component replaced by the given value."""
        if self._c_pos is None:
            raise KeyError("The vector does not have an c component.")
        return self.with_replaced(self._c_pos, new_c)

    def with_x(self: _T, new_x: int) -> _T:
        """Returns a new ND Vector with the x component replaced by the given value."""
        if self._x_pos is None:
            raise KeyError("The vector does not have an y component.")
        return self.with_replaced(self._x_pos, new_x)

    def with_y(self: _T, new_y: int) -> _T:
        """Returns a new ND Vector with the y component replaced by the given value."""
        if self._y_pos is None:
            raise KeyError("The vector does not have an y component.")
        return self.with_replaced(self._y_pos, new_y)

    def with_z(self: _T, new_z: int) -> _T:
        """Returns a new ND Vector with the z component replaced by the given value."""
        if self._z_pos is None:
            raise KeyError("The vector does not have an z component.")
        return self.with_replaced(self._z_pos, new_z)

    def with_xyz(self: _T, new_xyz: "Vec3Int") -> _T:
        """Returns a new ND Vector with the x, y and z components replaced by the given vector."""
        return self.with_x(new_xyz.x).with_y(new_xyz.y).with_z(new_xyz.z)

    def to_np(self) -> np.ndarray:
        """
        Returns the vector as a numpy array.
        """
        return np.array(self._data)

    def to_list(self) -> list[int]:
        """
        Returns the vector as a list.
        """
        return list(self._data)

    def to_tuple(self) -> tuple[int, ...]:
        """
        Returns the vector as a tuple.
        """
        return self._data

    def contains(self, needle: int) -> bool:
        """
        Checks if the vector contains a given element.
        """
        return any(element == needle for element in self)

    def is_positive(self, strictly_positive: bool = False) -> bool:
        """
        Checks if all elements in the vector are positive.

        Args:
            strictly_positive (bool): If True, checks if all elements are strictly positive.

        Returns:
            bool: True if all elements are positive, False otherwise.
        """
        if strictly_positive:
            return all(i > 0 for i in self)

        return all(i >= 0 for i in self)

    def is_uniform(self) -> bool:
        """
        Checks if all elements in the vector are the same.
        """
        first = self[0]
        return all(element == first for element in self)

    def _element_wise(
        self: _T, other: Union[int, "VecIntLike"], fn: Callable[[int, Any], int]
    ) -> _T:
        if isinstance(other, int):
            other_imported = VecInt.full(other, axes=self.axes)
        else:
            other_imported = VecInt(other, axes=self.axes)
            assert len(other_imported) == len(self), (
                f"{other} and {self} are not equally shaped."
            )
        return self.__class__(
            **{
                axis: fn(self[i], other_imported[i]) for i, axis in enumerate(self.axes)
            },
            axes=None,
        )

    # Note: When adding regular tuples the first tuple is extended with the second tuple.
    # For VecInt we want to add the elements at the same index.
    # Do not add VecInt to plain tuple! Hence the type:ignore)
    def __add__(self: _T, other: Union[int, "VecIntLike"]) -> _T:  # type: ignore[override]
        return self._element_wise(other, add)

    def __sub__(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        return self._element_wise(other, sub)

    # Note: When multiplying regular tuples with an int those are repeated,
    # which is a different behavior in the superclass! Hence the type:ignore.
    def __mul__(self: _T, other: Union[int, "VecIntLike"]) -> _T:  # type: ignore[override]
        return self._element_wise(other, mul)

    def __floordiv__(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        return self._element_wise(other, floordiv)

    def __mod__(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        return self._element_wise(other, mod)

    def __neg__(self: _T) -> _T:
        return self.__class__((-elem for elem in self), axes=self.axes)

    def ceildiv(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        """
        Returns a new VecInt with the ceil division of each element by the other.
        """
        return (self + other - 1) // other

    def pairmax(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        """
        Returns a new VecInt with the maximum of each pair of elements from the two vectors.
        """
        return self._element_wise(other, max)

    def pairmin(self: _T, other: Union[int, "VecIntLike"]) -> _T:
        """
        Returns a new VecInt with the minimum of each pair of elements from the two vectors.
        """
        return self._element_wise(other, min)

    def prod(self) -> int:
        """
        Returns the product of all elements in the vector.
        """
        return int(np.prod(self.to_np()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({','.join(str(element) for element in self)}, axes={self.axes})"

    def add_or_none(self: _T, other: Optional["VecInt"]) -> _T | None:
        """
        Adds two VecInts or returns None if the other is None.

        Args:
            other (VecInt | None): The other vector to add.

        Returns:
            VecInt | None: The sum of the two vectors or None if the other is None.
        """
        return None if other is None else self + other

    def moveaxis(self: _T, source: int | list[int], target: int | list[int]) -> _T:
        """
        Allows to move one element at index `source` to another index `target`. Similar to
        np.moveaxis, this is *not* a swap operation but instead it moves the specified
        source so that the other elements move when necessary.

        Args:
            source (int | list[int]): The index of the element to move.
            target (int | list[int]): The index where the element should be moved to.

        Returns:
            VecInt: A new vector with the moved element.
        """

        # Piggy-back on np.moveaxis by creating an auxiliary array where the indices 0, 1 and
        # 2 appear in the shape.
        indices = np.moveaxis(
            np.zeros(tuple(i for i in range(len(self)))), source, target
        ).shape
        arr = self.to_np()[np.array(indices)]
        axes = np.array(self.axes)[np.array(indices)]
        return self.__class__(arr, axes=axes)

    @classmethod
    def zeros(cls, axes: tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to 0.

        Args:
            axes (tuple[str, ...]): The axes of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((0 for _ in range(len(axes))), axes=axes)

    @classmethod
    def ones(cls, axes: tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to 1.

        Args:
            axes (tuple[str, ...]): The axes of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((1 for _ in range(len(axes))), axes=axes)

    @classmethod
    def full(cls, an_int: int, axes: tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to the same value.

        Args:
            an_int (int): The value to set all elements to.
            axes (tuple[str, ...]): The axes of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((an_int for _ in range(len(axes))), axes=axes)


VecIntLike: TypeAlias = VecInt | tuple[int, ...] | np.ndarray | Iterable[int]

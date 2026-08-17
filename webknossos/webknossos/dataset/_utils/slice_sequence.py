"""The base class for slice-by-slice image readers.

Only what the conversion path relies on: named axes declared by the reader,
with `bundle_axes` / `iter_axes` / `default_coords` selecting how they are
presented, plus random access to numbered slices and lazy slicing.

Every reader names its own axes, which is what lets `SlicedImageSource` stay a
single code path — inferring the order from a raw shape cannot tell a leading
sequence axis from a leading channel axis, and only the reader knows which it
has. Slices are plain `np.ndarray`s, since nothing downstream reads metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

GetSlice = Callable[..., np.ndarray]


def _resolve_index(index: int, length: int) -> int:
    i = int(index)
    if i < 0:
        i += length
    if not 0 <= i < length:
        raise IndexError(f"index {index} out of range for {length} slices")
    return i


def _resolve_indices(key: slice | Sequence[int] | np.ndarray, length: int) -> list[int]:
    if isinstance(key, slice):
        return list(range(*key.indices(length)))
    return [_resolve_index(i, length) for i in key]


class _SlicedView:
    """A lazy view on a subset of a `SliceSequence`'s slices: slicing,
    reversal (`[::-1]`), chaining, `len()` and iteration, reading nothing until
    iterated over. That is exactly what `copy_chunk_to_view` needs."""

    def __init__(self, source: SliceSequence, indices: Sequence[int]) -> None:
        self._source = source
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key: int | slice | Sequence[int]) -> Any:
        if isinstance(key, slice | Sequence | np.ndarray):
            selected = _resolve_indices(key, len(self._indices))
            return _SlicedView(self._source, [self._indices[i] for i in selected])
        return self._source.get_slice(
            self._indices[_resolve_index(key, len(self._indices))]
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in self._indices:
            yield self._source.get_slice(i)

    def __repr__(self) -> str:
        return f"<_SlicedView of {self._source!r} with {len(self)} slices>"


def _transpose(
    get_slice: GetSlice, expected_axes: Sequence[str], desired_axes: Sequence[str]
) -> GetSlice:
    if list(expected_axes) == list(desired_axes):
        return get_slice
    transposition = [list(expected_axes).index(a) for a in desired_axes]

    def get_slice_transposed(**ind: int) -> np.ndarray:
        return get_slice(**ind).transpose(transposition)

    return get_slice_transposed


def _drop(
    get_slice: GetSlice, expected_axes: Sequence[str], to_drop: Sequence[str]
) -> tuple[GetSlice, list[str]]:
    """Wraps `get_slice` so the axes in `to_drop` are indexed away, using the
    coordinate the caller passed for each of them."""
    # Sort axes in descending order so each np.take does not shift the
    # remaining axis positions.
    to_drop_inds = [list(expected_axes).index(a) for a in to_drop]
    order = list(np.argsort(to_drop_inds))
    axes = [to_drop_inds[i] for i in reversed(order)]
    dropped = [to_drop[i] for i in reversed(order)]
    result_axes = [a for a in expected_axes if a not in to_drop]

    def get_slice_dropped(**ind: int) -> np.ndarray:
        result = get_slice(**ind)
        for axis, name in zip(axes, dropped):
            result = np.take(result, ind[name], axis=axis)
        return result

    return get_slice_dropped, result_axes


def _adapt_get_slice(
    get_slice: GetSlice,
    source_axes: Sequence[str],
    result_axes: Sequence[str],
) -> GetSlice:
    """Adapts a reader's `get_slice` to return exactly `result_axes`.

    A reader declares one method covering every axis it can produce and callers
    ask for a subset, so this indexes away what was not asked for — using the
    coordinate passed for each — then transposes into the requested order.
    """
    source_axes = list(source_axes)
    result_axes = list(result_axes)
    if missing := [axis for axis in result_axes if axis not in source_axes]:
        raise ValueError(
            f"axes {missing!r} were requested but the reader's get_slice only "
            f"produces {source_axes!r}"
        )
    to_drop = [axis for axis in source_axes if axis not in result_axes]
    if to_drop:
        get_slice, source_axes = _drop(get_slice, source_axes, to_drop)
    return _transpose(get_slice, source_axes, result_axes)


class DefaultCoordsDict(dict[str, int]):
    """Coordinate dict that rejects axes the reader does not have."""

    def __init__(self) -> None:
        super().__init__()
        self.axes: list[str] = []

    def __setitem__(self, key: str, value: int) -> None:
        if key not in self.axes:
            raise ValueError(f"axis {key!r} does not exist")
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # Route through __setitem__ so the check applies here too.
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class SliceSequence(ABC):
    """A finite, randomly accessible sequence of slices over named axes.

    Subclasses declare their axes with `_init_axis(name, size)` and their one
    reader method with `_set_get_slice(method, axes)`, and define `pixel_type`.
    `__len__`, `slice_shape` and `get_slice` are derived from three attributes
    the caller sets:

    * `bundle_axes`: the axes making up one slice, in the order the returned
      array has them. Defaults to `["y", "x"]`.
    * `iter_axes`: the axes iterated over; the last varies fastest. Defaults
      to `[]`.
    * `default_coords`: the coordinate used for any axis in neither list.

    Slicing (`reader[1:4]`) returns a lazy view rather than reading anything.
    """

    # Consulted by `open_images()` to pick between readers that claim the same
    # extension; higher wins. 10 is the baseline, so a reader meant to take
    # precedence over the general-purpose ones sets something above it.
    class_priority: int = 10

    channels_are_colour: bool = False
    """Whether a "c" axis of this reader holds colour components rather than
    separate acquisitions. `SlicedImageSource` passes this on; see
    `ImageSource.channels_are_colour`."""

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        """The file extensions (without dot) this reader can open."""
        return set()

    def __init__(self) -> None:
        self._clear_axes()
        # The reader's own get_slice and the axes it returns; `bundle_axes`
        # narrows it to whatever the caller asked for.
        self._get_slice_source: tuple[GetSlice, tuple[str, ...]] | None = None

    def _clear_axes(self) -> None:
        self._sizes: dict[str, int] = {}
        self._default_coords = DefaultCoordsDict()
        self._iter_axes: list[str] = []
        self._bundle_axes: list[str] = ["y", "x"]
        self._get_slice_wrapped: GetSlice | None = None

    def _init_axis(self, name: str, size: int, default: int = 0) -> None:
        if name in self._sizes:
            raise ValueError(f"axis {name!r} already exists")
        self._sizes[name] = int(size)
        self._default_coords.axes = self.axes
        self._default_coords[name] = int(default)

    def _set_get_slice(self, method: GetSlice, axes: Iterable[str]) -> None:
        """Declares the reader method and the axes it returns."""
        self._get_slice_source = (method, tuple(axes))

    @property
    @abstractmethod
    def pixel_type(self) -> DTypeLike:
        """The dtype of the pixel values."""

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.pixel_type)

    def __len__(self) -> int:
        return int(np.prod([self._sizes[d] for d in self._iter_axes]))

    @property
    def slice_shape(self) -> tuple[int, ...]:
        """The shape of a single slice, as returned by `get_slice`."""
        return tuple(self._sizes[d] for d in self._bundle_axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self), *self.slice_shape)

    @property
    def axes(self) -> list[str]:
        return list(self._sizes)

    @property
    def ndim(self) -> int:
        return len(self._sizes)

    @property
    def sizes(self) -> dict[str, int]:
        return self._sizes

    @property
    def bundle_axes(self) -> list[str]:
        return self._bundle_axes[:]  # return a copy

    @bundle_axes.setter
    def bundle_axes(self, value: Iterable[str]) -> None:
        value = list(value)
        if invalid := [k for k in value if k not in self._sizes]:
            raise ValueError(f"axes {invalid!r} do not exist")

        for k in value:
            if k in self._iter_axes:
                self._iter_axes.remove(k)

        self._bundle_axes = value
        if self._get_slice_source is None:
            raise RuntimeError(
                "No reader method found. Declare one with _set_get_slice"
            )
        get_slice, source_axes = self._get_slice_source
        self._get_slice_wrapped = _adapt_get_slice(
            get_slice, source_axes, self._bundle_axes
        )

    @property
    def iter_axes(self) -> list[str]:
        return self._iter_axes[:]  # return a copy

    @iter_axes.setter
    def iter_axes(self, value: Iterable[str]) -> None:
        value = list(value)
        if invalid := [k for k in value if k not in self._sizes]:
            raise ValueError(f"axes {invalid!r} do not exist")

        for k in value:
            if k in self._bundle_axes:
                self._bundle_axes.remove(k)

        self._iter_axes = value

    @property
    def default_coords(self) -> DefaultCoordsDict:
        return self._default_coords

    @default_coords.setter
    def default_coords(self, value: dict[str, int]) -> None:
        self._default_coords.update(**value)

    def get_slice(self, i: int) -> np.ndarray:
        """Returns the slice at index `i`, shaped according to `bundle_axes`.

        `i` is interpreted as a flat index into `iter_axes`; axes in neither
        `iter_axes` nor `bundle_axes` use their `default_coords` value.
        """
        if i >= len(self):
            raise IndexError("index out of range")
        if self._get_slice_wrapped is None:
            self.bundle_axes = self.bundle_axes  # builds _get_slice_wrapped

        coords = dict(self.default_coords)
        coords.update(self.iter_coords(i))

        assert self._get_slice_wrapped is not None
        return np.asarray(self._get_slice_wrapped(**coords))

    def iter_coords(self, i: int) -> dict[str, int]:
        """Splits the flat index `i` into one coordinate per iter axis; the
        last axis varies fastest."""
        iter_sizes = [self._sizes[k] for k in self.iter_axes]
        # How much `i` must increase to advance each axis by one.
        iter_cumsizes = np.append(np.cumprod(iter_sizes[::-1])[-2::-1], 1)
        iter_coords = (i // iter_cumsizes) % iter_sizes
        return dict(zip(self.iter_axes, (int(c) for c in iter_coords)))

    def flat_index(self, coords: dict[str, int]) -> int:
        """The inverse of `iter_coords`: the index of the slice at the given
        iter-axis coordinates. Axes left out are taken to be at 0."""
        index = 0
        stride = 1
        for axis in reversed(self.iter_axes):
            index += coords.get(axis, 0) * stride
            stride *= self._sizes[axis]
        return index

    def __getitem__(self, key: int | slice | Sequence[int]) -> Any:
        if isinstance(key, slice | Sequence | np.ndarray):
            return _SlicedView(self, _resolve_indices(key, len(self)))
        return self.get_slice(_resolve_index(key, len(self)))

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self.get_slice(i)

    def close(self) -> None:
        """Release any resources held. Subclasses should call super()."""

    def __enter__(self) -> SliceSequence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        sizes = "".join(f"Axis '{a}' size: {s}\n" for a, s in self._sizes.items())
        return (
            f"<{type(self).__name__}>\nAxes: {self.ndim}\n{sizes}"
            f"Pixel Datatype: {self.dtype}"
        )

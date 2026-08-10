"""Base classes for slice-by-slice image readers.

These replace the parts of `pims` that this library used to depend on
(`pims.FramesSequence`, `pims.FramesSequenceND` and `slicerator.Slicerator`).
Only the behavior the conversion path actually relies on is implemented:

* random access to numbered frames, plus lazy slicing of a reader,
* named axes with `bundle_axes` / `iter_axes` / `default_coords`, and the
  resolver that maps a reader's registered `get_frame` onto whatever axes the
  caller asked for.

Unlike pims, frames are plain `np.ndarray`s — there is no `Frame` subclass and
no per-frame metadata, because nothing downstream reads it.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

GetFrame = Callable[..., np.ndarray]


class FrameSequence(ABC):
    """A finite, randomly accessible sequence of frames.

    Subclasses implement `__len__`, `get_frame` and the two shape/dtype
    properties. Slicing returns a lazy view rather than reading anything.
    """

    # Consulted by `open_images()` to pick between readers that claim the same
    # suffix; higher wins. The value mirrors pims' convention, where 10 was the
    # default for its built-in readers.
    class_priority: int = 10

    @classmethod
    def class_exts(cls) -> set[str]:
        """The file extensions (without dot) this reader can open."""
        return set()

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_frame(self, i: int) -> np.ndarray: ...

    @property
    @abstractmethod
    def frame_shape(self) -> tuple[int, ...]:
        """The shape of a single frame, as returned by `get_frame`."""

    @property
    @abstractmethod
    def pixel_type(self) -> DTypeLike:
        """The dtype of the pixel values."""

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.pixel_type)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self), *self.frame_shape)

    def __getitem__(self, key: int | slice | Sequence[int]) -> Any:
        if isinstance(key, slice | Sequence | np.ndarray):
            return _SlicedView(self, _resolve_indices(key, len(self)))
        return self.get_frame(_resolve_index(key, len(self)))

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self.get_frame(i)

    def close(self) -> None:
        """Release any resources held. Subclasses should call super()."""

    def __enter__(self) -> FrameSequence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}>\nLength: {len(self)} frames\n"
            f"Frame Shape: {self.frame_shape!r}\nPixel Datatype: {self.dtype}"
        )


def _resolve_index(index: int, length: int) -> int:
    i = int(index)
    if i < 0:
        i += length
    if not 0 <= i < length:
        raise IndexError(f"index {index} out of range for {length} frames")
    return i


def _resolve_indices(key: slice | Sequence[int] | np.ndarray, length: int) -> list[int]:
    if isinstance(key, slice):
        return list(range(*key.indices(length)))
    return [_resolve_index(i, length) for i in key]


class _SlicedView:
    """A lazy view on a subset of a `FrameSequence`'s frames.

    Replaces `slicerator.Slicerator`: it only has to support what
    `SlicedImages.copy_to_view` does — slicing, reversal (`[::-1]`), chained
    slicing of the result, `len()` and iteration — so no frame is read until it
    is actually iterated over.
    """

    def __init__(self, source: FrameSequence, indices: Sequence[int]) -> None:
        self._source = source
        self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key: int | slice | Sequence[int]) -> Any:
        if isinstance(key, slice | Sequence | np.ndarray):
            selected = _resolve_indices(key, len(self._indices))
            return _SlicedView(self._source, [self._indices[i] for i in selected])
        return self._source.get_frame(
            self._indices[_resolve_index(key, len(self._indices))]
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in self._indices:
            yield self._source.get_frame(i)

    def __repr__(self) -> str:
        return f"<_SlicedView of {self._source!r} with {len(self)} frames>"


def _transpose(
    get_frame: GetFrame, expected_axes: Sequence[str], desired_axes: Sequence[str]
) -> GetFrame:
    if list(expected_axes) == list(desired_axes):
        return get_frame
    transposition = [list(expected_axes).index(a) for a in desired_axes]

    def get_frame_transposed(**ind: int) -> np.ndarray:
        return get_frame(**ind).transpose(transposition)

    return get_frame_transposed


def _bundle(
    get_frame: GetFrame,
    expected_axes: Sequence[str],
    to_iter: Sequence[str],
    sizes: dict[str, int],
    dtype: DTypeLike,
) -> tuple[GetFrame, list[str]]:
    """Wraps `get_frame` so it also covers `to_iter`, by looping over those
    axes and stacking the results into one array."""
    bundled_axes = list(to_iter) + list(expected_axes)
    shape = [sizes[a] for a in bundled_axes]
    iter_shape = shape[: len(to_iter)]

    def get_frame_bundled(**ind: int) -> np.ndarray:
        result = np.empty(shape, dtype=dtype)
        for indices in itertools.product(*[range(s) for s in iter_shape]):
            ind.update(dict(zip(to_iter, indices)))
            result[indices] = get_frame(**ind)
        return result

    return get_frame_bundled, bundled_axes


def _drop(
    get_frame: GetFrame, expected_axes: Sequence[str], to_drop: Sequence[str]
) -> tuple[GetFrame, list[str]]:
    """Wraps `get_frame` so the axes in `to_drop` are indexed away, using the
    coordinate the caller passed for each of them."""
    # Sort axes in descending order so each np.take does not shift the
    # remaining axis positions.
    to_drop_inds = [list(expected_axes).index(a) for a in to_drop]
    order = list(np.argsort(to_drop_inds))
    axes = [to_drop_inds[i] for i in reversed(order)]
    dropped = [to_drop[i] for i in reversed(order)]
    result_axes = [a for a in expected_axes if a not in to_drop]

    def get_frame_dropped(**ind: int) -> np.ndarray:
        result = get_frame(**ind)
        for axis, name in zip(axes, dropped):
            result = np.take(result, ind[name], axis=axis)
        return result

    return get_frame_dropped, result_axes


def _make_get_frame(
    result_axes: Sequence[str],
    get_frame_dict: dict[tuple[str, ...], GetFrame],
    sizes: dict[str, int],
    dtype: DTypeLike,
) -> GetFrame:
    """Builds a function returning frames with exactly `result_axes`, out of
    the reader methods registered for other axis combinations.

    Preference order, matching pims: an exact match (possibly transposed), then
    reading no more data than needed (bundling extra axes), then avoiding
    iteration (dropping surplus axes), then the cheapest combination of both.
    """
    methods = list(get_frame_dict.keys())
    result_axes = list(result_axes)
    result_axes_set = set(result_axes)

    # A method that already returns the right axes, up to a transposition.
    for axes in methods:
        if len(set(axes) ^ result_axes_set) == 0:
            return _transpose(get_frame_dict[axes], axes, result_axes)

    # Otherwise axes have to be dropped or iterated over. Collect the cost of
    # both for every registered method.
    candidates = []
    for method in methods:
        axes_set = set(method)
        to_iter_set = result_axes_set - axes_set
        to_iter = [x for x in result_axes if x in to_iter_set]  # fix the order
        n_iter = int(np.prod([sizes[ax] for ax in to_iter]))
        to_drop = list(axes_set - result_axes_set)
        n_drop = int(np.prod([sizes[ax] for ax in to_drop]))
        candidates.append((method, to_iter, n_iter, to_drop, n_drop))

    # Prefer reading no superfluous data (n_drop == 0), fewest iterations first.
    candidates.sort(key=lambda x: x[2])
    for method, to_iter, _n_iter, _to_drop, n_drop in candidates:
        if n_drop > 0:
            continue
        get_frame, bundled_axes = _bundle(
            get_frame_dict[method], method, to_iter, sizes, dtype
        )
        return _transpose(get_frame, bundled_axes, result_axes)

    # Then prefer not iterating at all, dropping as few frames as possible.
    candidates.sort(key=lambda x: x[4])
    for method, _to_iter, n_iter, to_drop, _n_drop in candidates:
        if n_iter > 0:
            continue
        get_frame, after_drop = _drop(get_frame_dict[method], method, to_drop)
        return _transpose(get_frame, after_drop, result_axes)

    # Worst case: every method needs both. Take the fewest dropped frames, and
    # among those the fewest iterations.
    candidates.sort(key=lambda x: (x[2], x[4]))
    method, to_iter, _n_iter, to_drop, _n_drop = candidates[0]
    get_frame, after_drop = _drop(get_frame_dict[method], method, to_drop)
    get_frame, after_bundle = _bundle(get_frame, after_drop, to_iter, sizes, dtype)
    return _transpose(get_frame, after_bundle, result_axes)


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


class NDFrameSequence(FrameSequence):
    """A `FrameSequence` whose data has arbitrary named axes.

    Subclasses declare their axes with `_init_axis(name, size)` and register at
    least one reader method with `_register_get_frame(method, axes)`; they only
    need to additionally define `pixel_type`. Everything below — `__len__`,
    `frame_shape`, `get_frame` — is derived from three attributes the caller
    sets:

    * `bundle_axes`: the axes making up one frame, in the order the returned
      array has them. Defaults to `["y", "x"]`.
    * `iter_axes`: the axes iterated over by the sequence; the last one varies
      fastest. Defaults to `[]`.
    * `default_coords`: the coordinate used for any axis in neither list.
    """

    def __init__(self) -> None:
        self._clear_axes()
        self._get_frame_dict: dict[tuple[str, ...], GetFrame] = {}

    def _clear_axes(self) -> None:
        self._sizes: dict[str, int] = {}
        self._default_coords = DefaultCoordsDict()
        self._iter_axes: list[str] = []
        self._bundle_axes: list[str] = ["y", "x"]
        self._get_frame_wrapped: GetFrame | None = None

    def _init_axis(self, name: str, size: int, default: int = 0) -> None:
        if name in self._sizes:
            raise ValueError(f"axis {name!r} already exists")
        self._sizes[name] = int(size)
        self._default_coords.axes = self.axes
        self._default_coords[name] = int(default)

    def _register_get_frame(self, method: GetFrame, axes: Iterable[str]) -> None:
        self._get_frame_dict[tuple(axes)] = method

    def __len__(self) -> int:
        return int(np.prod([self._sizes[d] for d in self._iter_axes]))

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return tuple(self._sizes[d] for d in self._bundle_axes)

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
        if not self._get_frame_dict:
            raise RuntimeError(
                "No reader methods found. Register a reader method with "
                "_register_get_frame"
            )
        self._get_frame_wrapped = _make_get_frame(
            self._bundle_axes, self._get_frame_dict, self.sizes, self.pixel_type
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

    def get_frame(self, i: int) -> np.ndarray:
        """Returns the frame at index `i`, shaped according to `bundle_axes`.

        `i` is interpreted as a flat index into `iter_axes`; axes in neither
        `iter_axes` nor `bundle_axes` use their `default_coords` value.
        """
        if i >= len(self):
            raise IndexError("index out of range")
        if self._get_frame_wrapped is None:
            self.bundle_axes = self.bundle_axes  # builds _get_frame_wrapped

        coords = dict(self.default_coords)

        # How much `i` has to increase for each iter axis' coordinate to
        # increase by one; the last axis varies fastest.
        iter_sizes = [self._sizes[k] for k in self.iter_axes]
        iter_cumsizes = np.append(np.cumprod(iter_sizes[::-1])[-2::-1], 1)
        iter_coords = (i // iter_cumsizes) % iter_sizes
        coords.update(dict(zip(self.iter_axes, (int(c) for c in iter_coords))))

        assert self._get_frame_wrapped is not None
        return np.asarray(self._get_frame_wrapped(**coords))

    def __repr__(self) -> str:
        sizes = "".join(f"Axis '{a}' size: {s}\n" for a, s in self._sizes.items())
        return (
            f"<{type(self).__name__}>\nAxes: {self.ndim}\n{sizes}"
            f"Pixel Datatype: {self.dtype}"
        )

"""Shared base for `ChunkedImageSource`s backed by a tensorstore driver.

Zarr, N5 and neuroglancer precomputed are all read through tensorstore, so once
a subclass has resolved an open spec (`_ts_spec`) and knows which physical
array dimension holds which axis (`_axes`), reading a box is identical
regardless of driver — that shared logic lives here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorstore as ts
from upath import UPath

from ...utils import is_remote_path
from .._utils.tensorstore_helpers import TS_CONTEXT
from ..errors import UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions

# Axis names, in the canonical order a `_read_source_box` result uses (t and c
# are squeezed out again in that order once selected/sliced).
_CANONICAL_AXIS_ORDER = ("t", "c", "z", "y", "x")
_KNOWN_AXES = frozenset(_CANONICAL_AXIS_ORDER)
_AXIS_ALIASES = {"channel": "c", "time": "t"}


def normalize_axis(label: str, *, path: UPath) -> str:
    """Normalizes an axis label (e.g. from Zarr `dimension_names` or OME
    `axes` metadata) to one of `t`/`c`/`z`/`y`/`x`."""
    axis = _AXIS_ALIASES.get(label.lower(), label.lower())
    if axis not in _KNOWN_AXES:
        raise UnsupportedImageDataError(
            f"Cannot place axis {label!r} of {path} — only t/c/z/y/x axes are "
            "supported.",
            path=path,
        )
    return axis


def guess_axes(
    ndim: int,
    *,
    axis_labels: list[str] | None,
    path: UPath,
) -> tuple[str, ...]:
    """
    Maps each physical array dimension (position i) to one of
    `t`/`c`/`z`/`y`/`x`.

    `axis_labels` (tensorstore `domain.labels`, Zarr v3 `dimension_names`, or
    axis names derived from OME `axes` metadata) are used directly when given
    and non-empty; otherwise falls back to the OME-NGFF canonical positional
    convention, right-aligned: 2D->(y,x), 3D->(z,y,x), 4D->(c,z,y,x),
    5D->(t,c,z,y,x).
    """
    if axis_labels and all(axis_labels):
        axes = tuple(normalize_axis(label, path=path) for label in axis_labels)
        if len(set(axes)) != len(axes):
            raise UnsupportedImageDataError(
                f"Duplicate axes {axes} for {path}.", path=path
            )
        return axes

    if ndim == 2:
        return ("y", "x")
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 4:
        return ("c", "z", "y", "x")
    if ndim == 5:
        return ("t", "c", "z", "y", "x")
    raise UnsupportedImageDataError(
        f"Cannot place the {ndim} axes of {path} — only 2 to 5 dimensional "
        "arrays are supported.",
        path=path,
    )


class TensorStoreChunkedImageSource(ChunkedImageSource):
    """
    ChunkedImageSource for formats tensorstore reads directly (Zarr, N5,
    neuroglancer precomputed). Subclasses resolve `self._ts_spec` (the
    tensorstore open spec, already pointing at the chosen resolution level for
    multiscale sources) and `self._axes` (the physical dimension order, from
    `t`/`c`/`z`/`y`/`x`) in `__init__`, together with the usual
    `ChunkedImageSource` fields via `compute_channel_selection()`.
    """

    _ts_spec: dict[str, Any]
    _axes: tuple[str, ...]
    # Set by subclasses via compute_channel_selection(), same as
    # CziImageSource/ImsImageSource.
    _first_n_channels: int | None

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        # Tensorstore itself can read s3://, gs:// and http(s):// just fine,
        # but conversion from remote paths isn't supported for these formats
        # yet — restricted here, once, for every subclass.
        if is_remote_path(path):
            raise ValueError(
                f"Cannot open {path}. Remote paths (s3://, gs://, http(s)://) "
                "are not supported yet; the path must be a local file path."
            )

    def _open_array(self) -> ts.TensorStore:
        # Reopened on every call rather than cached: chunks are read from
        # separate worker processes, so no tensorstore handle can cross that
        # boundary (mirrors MrcImageSource's per-call mmap reopen).
        return ts.open(
            self._ts_spec, open=True, context=TS_CONTEXT, recheck_cached="open"
        ).result()

    def _read_source_box(
        self,
        *,
        timepoint: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        if self._channel is not None:
            channels_to_read = [self._channel]
        elif self._first_n_channels is not None:
            channels_to_read = list(range(self._first_n_channels))
        else:
            channels_to_read = list(range(self.num_channels))

        has_t = "t" in self._axes
        has_c = "c" in self._axes
        has_z = "z" in self._axes

        # The order to transpose the read result into, and which of those
        # positions are the size-1 t/c axes to squeeze back out afterwards —
        # computed once since it does not depend on the channel being read.
        present_axes = [axis for axis in _CANONICAL_AXIS_ORDER if axis in self._axes]
        permutation = [self._axes.index(axis) for axis in present_axes]
        squeeze_positions = tuple(
            i for i, axis in enumerate(present_axes) if axis in ("t", "c")
        )

        array = self._open_array()
        slabs = []
        for channel_index in channels_to_read:
            axis_to_slice: dict[str, slice] = {"y": y, "x": x}
            if has_t:
                axis_to_slice["t"] = slice(timepoint, timepoint + 1)
            if has_z:
                axis_to_slice["z"] = z
            if has_c:
                axis_to_slice["c"] = slice(channel_index, channel_index + 1)
            index = tuple(axis_to_slice[axis] for axis in self._axes)

            block = np.asarray(array[index].read().result())
            block = block.transpose(permutation)
            if squeeze_positions:
                block = block.squeeze(axis=squeeze_positions)
            if not has_z:
                block = block[np.newaxis]  # add a size-1 z axis
            slabs.append(block)
        return np.stack(slabs, axis=0)  # (c, z, y, x)

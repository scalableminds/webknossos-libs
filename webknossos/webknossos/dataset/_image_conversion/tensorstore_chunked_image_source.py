"""Shared base for `ChunkedImageSource`s backed by a tensorstore driver.

Zarr, N5 and neuroglancer precomputed are all read through tensorstore, so once
a subclass has resolved an open spec (`_ts_spec`) and knows which physical
array dimension holds which axis (`_axis_roles`), reading a box is identical
regardless of driver — that shared logic lives here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorstore as ts
from upath import UPath

from .._utils.tensorstore_helpers import TS_CONTEXT
from ..errors import UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource

# Axis role names, in the canonical order a `_read_source_box` result uses
# (t and c are squeezed out again in that order once selected/sliced).
_CANONICAL_ROLE_ORDER = ("t", "c", "z", "y", "x")
_KNOWN_ROLES = frozenset(_CANONICAL_ROLE_ORDER)
_ROLE_ALIASES = {"channel": "c", "time": "t"}


def normalize_axis_role(label: str, *, path: UPath) -> str:
    """Maps an axis label (e.g. from Zarr `dimension_names` or OME `axes`
    metadata) to one of the roles in `{"t", "c", "z", "y", "x"}`."""
    role = _ROLE_ALIASES.get(label.lower(), label.lower())
    if role not in _KNOWN_ROLES:
        raise UnsupportedImageDataError(
            f"Cannot place axis {label!r} of {path} — only t/c/z/y/x axes are "
            "supported.",
            path=path,
        )
    return role


def guess_axis_roles(
    ndim: int,
    *,
    axis_labels: list[str] | None,
    path: UPath,
) -> tuple[str, ...]:
    """
    Maps each physical array dimension (position i) to a role in
    `{"t", "c", "z", "y", "x"}`.

    `axis_labels` (tensorstore `domain.labels`, Zarr v3 `dimension_names`, or
    axis names derived from OME `axes` metadata) are used directly when given
    and non-empty; otherwise falls back to the OME-NGFF canonical positional
    convention, right-aligned: 2D->(y,x), 3D->(z,y,x), 4D->(c,z,y,x),
    5D->(t,c,z,y,x).
    """
    if axis_labels and all(axis_labels):
        roles = tuple(normalize_axis_role(label, path=path) for label in axis_labels)
        if len(set(roles)) != len(roles):
            raise UnsupportedImageDataError(
                f"Duplicate axis roles {roles} for {path}.", path=path
            )
        return roles

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
    multiscale sources) and `self._axis_roles` (the physical dimension order,
    as roles from `{"t", "c", "z", "y", "x"}`) in `__init__`, together with the
    usual `ChunkedImageSource` fields via `compute_channel_selection()`.
    """

    _ts_spec: dict[str, Any]
    _axis_roles: tuple[str, ...]
    # Set by subclasses via compute_channel_selection(), same as
    # CziImageSource/ImsImageSource.
    _first_n_channels: int | None

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

        has_t = "t" in self._axis_roles
        has_c = "c" in self._axis_roles
        has_z = "z" in self._axis_roles

        # The order to transpose the read result into, and which of those
        # positions are the size-1 t/c axes to squeeze back out afterwards —
        # computed once since it does not depend on the channel being read.
        present_roles = [
            role for role in _CANONICAL_ROLE_ORDER if role in self._axis_roles
        ]
        permutation = [self._axis_roles.index(role) for role in present_roles]
        squeeze_positions = tuple(
            i for i, role in enumerate(present_roles) if role in ("t", "c")
        )

        array = self._open_array()
        slabs = []
        for channel_index in channels_to_read:
            role_to_slice: dict[str, slice] = {"y": y, "x": x}
            if has_t:
                role_to_slice["t"] = slice(timepoint, timepoint + 1)
            if has_z:
                role_to_slice["z"] = z
            if has_c:
                role_to_slice["c"] = slice(channel_index, channel_index + 1)
            index = tuple(role_to_slice[role] for role in self._axis_roles)

            block = np.asarray(array[index].read().result())
            block = block.transpose(permutation)
            if squeeze_positions:
                block = block.squeeze(axis=squeeze_positions)
            if not has_z:
                block = block[np.newaxis]  # add a size-1 z axis
            slabs.append(block)
        return np.stack(slabs, axis=0)  # (c, z, y, x)

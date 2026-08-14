"""Shared base for `ChunkedImageSource` subclasses backed by a
tensorstore-openable array: Zarr (v2/v3), N5, and neuroglancer precomputed.

Once a tensorstore open spec and a mapping from logical axis role to physical
dimension index are known, reading a box is identical regardless of driver —
this class implements that once. Format-specific subclasses only resolve the
open spec (already pointing at the chosen resolution level for a multiscale
source) and the axis roles from each format's own metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorstore as ts

from ...utils import call_with_retries
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions, compute_channel_selection
from .tensorstore_helpers import TS_CONTEXT

# A physical dimension's role. "t" and "c" appear at most once each; "z"/"y"/"x"
# are the spatial axes _read_source_box always receives.
AxisRole = str

_CANONICAL_AXIS_ORDER: tuple[AxisRole, ...] = ("t", "c", "z", "y", "x")
"""OME-NGFF's canonical axis order, used to guess roles for arrays that carry
no axis names at all (a bare Zarr v2 `.zarray`, a plain N5 dataset)."""

_LABEL_TO_ROLE: dict[str, AxisRole] = {
    "x": "x",
    "y": "y",
    "z": "z",
    "c": "c",
    "channel": "c",
    "t": "t",
    "time": "t",
}


def resolve_axis_roles(labels: Sequence[str]) -> tuple[AxisRole, ...] | None:
    """Maps tensorstore dimension labels (Zarr v3 `dimension_names`) to axis
    roles, or None if any dimension is unlabeled (tensorstore reports `""` for
    those) or its label isn't recognized — the caller then falls back to
    `positional_axis_roles`."""
    if not labels or any(label == "" for label in labels):
        return None
    roles = [_LABEL_TO_ROLE.get(label.lower(), "") for label in labels]
    if "" in roles or len(set(roles)) != len(roles):
        # An unrecognized label, or the same role claimed twice, is not a
        # mapping worth trusting.
        return None
    return tuple(roles)


def positional_axis_roles(ndim: int) -> tuple[AxisRole, ...]:
    """OME-NGFF's canonical `(t, c, z, y, x)` axis order, right-aligned and
    trimmed to `ndim`, for arrays with no axis names to go by: 2D -> (y, x),
    3D -> (z, y, x), 4D -> (c, z, y, x), 5D -> (t, c, z, y, x)."""
    if not 1 <= ndim <= len(_CANONICAL_AXIS_ORDER):
        raise ValueError(
            f"Cannot guess axis roles for a {ndim}-dimensional array without "
            f"axis names; only 1-{len(_CANONICAL_AXIS_ORDER)} dimensions are supported."
        )
    return _CANONICAL_AXIS_ORDER[len(_CANONICAL_AXIS_ORDER) - ndim :]


class _TensorStoreChunkedImageSource(ChunkedImageSource):
    """
    `_ts_spec` is the resolved tensorstore open spec and `_axis_order` says
    which physical dimension holds which role; subclasses set both (plus the
    usual `ChunkedImageSource` fields) in `__init__`, typically via
    `_finish_init_from_array`.

    The array handle is reopened on every `_read_source_box()` call rather
    than cached: chunks may run in separate processes, and no open handle may
    cross that boundary (see `ChunkedImageSource._read_source_box`'s
    contract).
    """

    _ts_spec: dict[str, object]
    _axis_order: tuple[AxisRole, ...]

    def _open_array(self) -> ts.TensorStore:
        driver = self._ts_spec.get("driver")
        return call_with_retries(
            lambda: ts.open(
                self._ts_spec, open=True, context=TS_CONTEXT, recheck_cached="open"
            ).result(),
            description=f"Opening {driver} array at {self.path}",
        )

    def _finish_init_from_array(
        self,
        array: ts.TensorStore,
        axis_order: tuple[AxisRole, ...],
        options: ReadOptions,
    ) -> list[int] | None:
        """Sets dtype/_x/_y/_z/_t/num_channels/_channel/_first_n_channels/
        _include_t_axis/_fixed_timepoint/_axis_order from an opened array and
        its axis-role mapping. Returns the possible channel indices (see
        `compute_channel_selection`), for the caller to fold into
        `get_possible_layers()`."""
        assert len(axis_order) == array.ndim, (
            f"Resolved {len(axis_order)} axis roles for a {array.ndim}-dimensional array."
        )
        dims = dict(zip(axis_order, array.shape, strict=True))
        self._axis_order = axis_order
        self.dtype = array.dtype.numpy_dtype
        self._x = dims["x"]
        self._y = dims["y"]
        self._z = dims.get("z", 1)
        self._t = dims.get("t", 1)
        raw_num_channels = dims.get("c", 1)

        (
            self.num_channels,
            self._channel,
            self._first_n_channels,
            possible_channels,
        ) = compute_channel_selection(raw_num_channels, options.channel)

        self._include_t_axis = self._t > 1
        self._fixed_timepoint = None if self._include_t_axis else 0
        return possible_channels

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

        array = self._open_array()
        values: dict[AxisRole, object] = {"t": timepoint, "z": z, "y": y, "x": x}

        # Indexing with an int for "t"/"c" drops that axis and keeps the rest
        # in their original relative order, so after the read the remaining
        # axes are exactly the spatial roles in self._axis_order, in that
        # order — not necessarily (z, y, x). A role missing from the source
        # entirely (e.g. no "z" in a 2D array) gets a synthesized size-1 axis,
        # appended here and permuted into place below, so every read comes
        # back as (z, y, x) regardless of the source's own axis order.
        present = [role for role in self._axis_order if role in ("z", "y", "x")]
        missing = [role for role in ("z", "y", "x") if role not in present]
        result_roles = present + missing
        permutation = [result_roles.index(role) for role in ("z", "y", "x")]

        slabs = []
        for channel in channels_to_read:
            values["c"] = channel
            index = tuple(values[role] for role in self._axis_order)

            def _read(index: tuple[object, ...] = index) -> np.ndarray:
                return np.asarray(array[index].read().result())

            slab = call_with_retries(
                _read, description=f"Reading {self._ts_spec.get('driver')} array"
            )
            for _ in missing:
                slab = slab[..., np.newaxis]
            slabs.append(slab.transpose(permutation))
        return np.stack(slabs, axis=0)  # (c, z, y, x)

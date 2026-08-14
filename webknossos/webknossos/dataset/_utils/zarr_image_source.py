"""ChunkedImageSource for plain Zarr arrays and OME-Zarr multiscale groups.

Handles both Zarr v2 (`.zarray`/`.zgroup`/`.zattrs`) and Zarr v3 (`zarr.json`)
stores, and both plain single-resolution arrays and OME-NGFF multiscale
groups (0.4 on Zarr v2, 0.5 on Zarr v3). A multiscale group is a Zarr group
whose `multiscales` metadata lists one plain array per resolution level; this
reader always resolves to a plain array before opening it, so from there on a
multiscale group and a directly-given array are read identically.
"""

from __future__ import annotations

import json
import math
from typing import Any

from upath import UPath

from ..defaults import (
    ZARR_JSON_FILE_NAME,
    ZARRAY_FILE_NAME,
    ZATTRS_FILE_NAME,
    ZGROUP_FILE_NAME,
)
from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from .image_source import ReadOptions
from .image_source_registry import register_chunked_reader
from .tensorstore_chunked_source import (
    AxisRole,
    _TensorStoreChunkedImageSource,
    positional_axis_roles,
    resolve_axis_roles,
)
from .tensorstore_helpers import _make_kvstore


def _read_json(path: UPath) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def _dataset_scale(dataset: dict[str, Any]) -> list[float] | None:
    for transform in dataset.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            return [float(v) for v in transform["scale"]]
    return None


def _axis_roles_from_ome_axes(
    axes: list[dict[str, Any]], *, path: UPath
) -> tuple[AxisRole, ...]:
    roles: list[AxisRole] = []
    for axis in axes:
        name = str(axis.get("name", "")).lower()
        axis_type = axis.get("type")
        if name in ("x", "y", "z"):
            roles.append(name)
        elif axis_type == "channel" or name == "c":
            roles.append("c")
        elif axis_type == "time" or name == "t":
            roles.append("t")
        else:
            raise UnsupportedImageDataError(
                f"The OME-Zarr axis {axis!r} in {path} is not a channel, time, "
                "or x/y/z space axis, which this reader cannot place.",
                path=path,
            )
    return tuple(roles)


@register_chunked_reader
class ZarrImageSource(_TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for plain Zarr arrays (v2 or v3) and OME-Zarr
    multiscale groups (NGFF 0.4 on Zarr v2, NGFF 0.5 on Zarr v3). A multiscale
    group is resolved to its finest-resolution array by default; pass
    `format_options={"scale": <index>}` (0 = finest) to pick another.
    """

    @classmethod
    def class_exts(cls) -> set[str]:
        return {"zarr"}

    @classmethod
    def probe_directory(cls, path: UPath) -> bool:
        return (
            (path / ZARR_JSON_FILE_NAME).is_file()
            or (path / ZARRAY_FILE_NAME).is_file()
            or (path / ZGROUP_FILE_NAME).is_file()
        )

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        self._num_scale_levels = 1

        try:
            resolved_path, driver, axis_roles = self._resolve_array(path, options)
            self._ts_spec = {"driver": driver, "kvstore": _make_kvstore(resolved_path)}
            array = self._open_array()
            if axis_roles is None:
                axis_roles = resolve_axis_roles(
                    array.domain.labels
                ) or positional_axis_roles(array.ndim)
            possible_channels = self._finish_init_from_array(array, axis_roles, options)
        except (
            CorruptImageError,
            UnsupportedImageFormatError,
            UnsupportedImageDataError,
        ):
            raise
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open Zarr store {path}. It is likely corrupted or not "
                "a valid Zarr array or OME-Zarr multiscale group.",
                path=path,
            ) from e

        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels
        if self._num_scale_levels > 1:
            self._possible_layers["scale"] = list(range(self._num_scale_levels))

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        return self._possible_layers or None

    def _resolve_array(
        self, path: UPath, options: ReadOptions
    ) -> tuple[UPath, str, tuple[AxisRole, ...] | None]:
        """Returns (path to the plain array to open, tensorstore driver, axis
        roles already known from OME multiscale `axes` metadata — or None to
        derive them from the array itself once it is opened)."""
        zarr_json_path = path / ZARR_JSON_FILE_NAME
        zarray_path = path / ZARRAY_FILE_NAME
        zgroup_path = path / ZGROUP_FILE_NAME

        if zarr_json_path.is_file():
            metadata = _read_json(zarr_json_path)
            node_type = metadata.get("node_type")
            if node_type == "array":
                return path, "zarr3", None
            if node_type == "group":
                return self._resolve_multiscale(
                    path, metadata.get("attributes", {}) or {}, options, is_v3=True
                )
            raise CorruptImageError(
                f"{zarr_json_path} has an unrecognized node_type {node_type!r}.",
                path=path,
            )
        if zarray_path.is_file():
            # Zarr v2 has no equivalent of v3's dimension_names, so a bare
            # .zarray always falls back to positional axis guessing.
            return path, "zarr", None
        if zgroup_path.is_file():
            zattrs_path = path / ZATTRS_FILE_NAME
            attributes = _read_json(zattrs_path) if zattrs_path.is_file() else {}
            return self._resolve_multiscale(path, attributes, options, is_v3=False)
        raise CorruptImageError(
            f"{path} is not a valid Zarr store (no zarr.json/.zarray/.zgroup found).",
            path=path,
        )

    def _resolve_multiscale(
        self,
        group_path: UPath,
        attributes: dict[str, Any],
        options: ReadOptions,
        *,
        is_v3: bool,
    ) -> tuple[UPath, str, tuple[AxisRole, ...]]:
        multiscales = (
            attributes.get("ome", attributes).get("multiscales")
            if is_v3
            else attributes.get("multiscales")
        )
        if not multiscales:
            raise UnsupportedImageFormatError(
                f"{group_path} is a Zarr group, but not an OME-Zarr multiscale "
                "group (no `multiscales` metadata found).",
                path=group_path,
            )
        # Only the first multiscale entry is supported, matching every other
        # OME-Zarr reader's convention.
        multiscale = multiscales[0]
        datasets = multiscale.get("datasets") or []
        if not datasets:
            raise CorruptImageError(
                f"The OME-Zarr multiscale group at {group_path} lists no datasets.",
                path=group_path,
            )
        axis_roles = _axis_roles_from_ome_axes(
            multiscale.get("axes", []), path=group_path
        )
        spatial_indices = [i for i, role in enumerate(axis_roles) if role in "xyz"]

        def resolution_key(dataset: dict[str, Any]) -> float:
            scale = _dataset_scale(dataset)
            if scale is None:
                return 1.0
            return math.prod(scale[i] for i in spatial_indices)

        # Sorted finest (smallest scale factor) first — the order
        # get_possible_layers()["scale"] promises, and what "scale" in
        # format_options indexes into. Does not assume `datasets` is already
        # ordered this way, since the spec does not strictly require it.
        scale_order = sorted(
            range(len(datasets)), key=lambda i: resolution_key(datasets[i])
        )
        self._num_scale_levels = len(datasets)

        requested_scale = options.format_option("scale")
        if requested_scale is not None:
            assert 0 <= requested_scale < len(datasets), (
                f"Selected scale {requested_scale} (0-indexed), but the OME-Zarr "
                f"multiscale group at {group_path} only has {len(datasets)} levels."
            )
            chosen_index = scale_order[requested_scale]
        else:
            chosen_index = scale_order[0]

        dataset_path = datasets[chosen_index]["path"]
        driver = "zarr3" if is_v3 else "zarr"
        return group_path / dataset_path, driver, axis_roles

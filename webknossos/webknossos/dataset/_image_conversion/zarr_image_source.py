"""Reader for plain Zarr arrays and OME-Zarr multiscale groups.

Handles both Zarr v2 (`.zarray`/`.zgroup`/`.zattrs`) and Zarr v3 (`zarr.json`)
stores, and both a bare array and an OME-Zarr multiscale group wrapping one —
for a multiscale group, only the first `multiscales` entry is considered, and
the finest resolution level is opened by default. `ReadOptions.format_options`
`"scale"` picks a different level (0 = finest), the same way `czi_channel`
picks a CZI acquisition channel.
"""

from __future__ import annotations

import json
from typing import Any

import tensorstore as ts
from upath import UPath

from .._utils.tensorstore_helpers import TS_CONTEXT, _make_kvstore
from ..defaults import (
    ZARR_JSON_FILE_NAME,
    ZARRAY_FILE_NAME,
    ZATTRS_FILE_NAME,
    ZGROUP_FILE_NAME,
)
from ..errors import CorruptImageError
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_image_source
from .ome_zarr_multiscale import resolve_ome_multiscale
from .tensorstore_chunked_image_source import TensorStoreChunkedImageSource, guess_axes


def _read_json(path: UPath, *, path_for_errors: UPath) -> Any:
    try:
        return json.loads(path.read_bytes())
    except Exception as e:
        raise CorruptImageError(
            f"Cannot read {path}. It is likely corrupted or not valid JSON.",
            path=path_for_errors,
        ) from e


@register_chunked_image_source
class ZarrImageSource(TensorStoreChunkedImageSource):
    """
    ChunkedImageSource for plain Zarr arrays (v2 `.zarray`, v3 `zarr.json`
    array nodes) and OME-Zarr multiscale groups (NGFF 0.4 on Zarr v2, NGFF 0.5
    on Zarr v3). For a multiscale group, the finest resolution level is opened
    by default; only the first `multiscales` entry is considered.
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
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
        self._possible_layers: dict[str, list[int]] = {}

        resolved_path, axis_name_hint = self._resolve_array_path(path, options)

        try:
            driver = (
                "zarr3" if (resolved_path / ZARR_JSON_FILE_NAME).is_file() else "zarr"
            )
            self._ts_spec = {"driver": driver, "kvstore": _make_kvstore(resolved_path)}
            array = ts.open(
                self._ts_spec, open=True, context=TS_CONTEXT, recheck_cached="open"
            ).result()
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open the Zarr array at {resolved_path} (from {path}). "
                "It is likely corrupted or not a valid Zarr array.",
                path=path,
            ) from e

        domain_labels = list(array.domain.labels)
        axis_labels = axis_name_hint or (domain_labels if any(domain_labels) else None)
        self._axes = guess_axes(len(array.shape), axis_labels=axis_labels, path=path)

        shape = array.domain.exclusive_max
        axis_to_size = dict(zip(self._axes, shape))
        self._x = axis_to_size.get("x", 1)
        self._y = axis_to_size.get("y", 1)
        self._z = axis_to_size.get("z", 1)
        raw_num_channels = axis_to_size.get("c", 1)
        self.dtype = array.dtype.numpy_dtype

        t = axis_to_size.get("t", 1)
        self._t = t
        self._include_t_axis = t > 1
        self._fixed_timepoint = None if self._include_t_axis else 0

        (
            self.num_channels,
            self._channel,
            self._first_n_channels,
            possible_channels,
        ) = compute_channel_selection(raw_num_channels, options.channel)
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    def _resolve_array_path(
        self, path: UPath, options: ReadOptions
    ) -> tuple[UPath, list[str] | None]:
        """
        Resolves `path` to the Zarr array to actually open: itself, if it is
        already a plain array, or — for an OME-Zarr multiscale group — the
        dataset at the requested `scale` rank (0 = finest, the default).
        Also returns axis names derived from the group's OME `axes` metadata
        when resolving out of a group, since a v2 sub-array carries no axis
        names of its own.
        """
        zarr_json_path = path / ZARR_JSON_FILE_NAME
        zarray_path = path / ZARRAY_FILE_NAME
        zgroup_path = path / ZGROUP_FILE_NAME

        if zarr_json_path.is_file():
            metadata = _read_json(zarr_json_path, path_for_errors=path)
            node_type = metadata.get("node_type")
            if node_type == "array":
                return path, None
            if node_type == "group":
                attributes = metadata.get("attributes", {})
                ome = attributes.get("ome", attributes)
                return self._resolve_ome_multiscale(path, ome, options)
            raise CorruptImageError(
                f"{zarr_json_path} has an unexpected node_type {node_type!r}.",
                path=path,
            )

        if zarray_path.is_file():
            return path, None

        if zgroup_path.is_file():
            zattrs_path = path / ZATTRS_FILE_NAME
            attributes = (
                _read_json(zattrs_path, path_for_errors=path)
                if zattrs_path.is_file()
                else {}
            )
            return self._resolve_ome_multiscale(path, attributes, options)

        raise CorruptImageError(
            f"{path} is not a valid Zarr store (no {ZARR_JSON_FILE_NAME}/"
            f"{ZARRAY_FILE_NAME}/{ZGROUP_FILE_NAME} found).",
            path=path,
        )

    def _resolve_ome_multiscale(
        self, path: UPath, attributes: dict[str, Any], options: ReadOptions
    ) -> tuple[UPath, list[str] | None]:
        multiscale = resolve_ome_multiscale(attributes, path=path)

        rank = options.format_option("scale")
        rank = 0 if rank is None else rank
        if not (0 <= rank < len(multiscale.dataset_paths)):
            raise ValueError(
                f"scale {rank} does not exist in {path}. Available: "
                f"{list(range(len(multiscale.dataset_paths)))}."
            )

        if len(multiscale.dataset_paths) > 1:
            self._possible_layers["scale"] = list(range(len(multiscale.dataset_paths)))

        resolved_path = path / multiscale.dataset_paths[rank]
        axis_names = list(multiscale.axis_names) if multiscale.axis_names else None
        return resolved_path, axis_names

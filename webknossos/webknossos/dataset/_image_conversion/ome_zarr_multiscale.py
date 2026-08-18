"""Shared OME-Zarr multiscale group parsing: ranking a group's `datasets` by
resolution and naming its axes. Used by both `ZarrImageSource` (a multiscale
group stored as a directory) and `OzxImageSource` (one zipped into an `.ozx`
archive) — the `multiscales` metadata itself is identical either way, only
how the resolved dataset path is turned into something openable differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from upath import UPath

from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from .tensorstore_chunked_image_source import normalize_axis

_SUPPORTED_OME_VERSIONS = ("0.4", "0.5")


def _scale_transform(dataset: dict[str, Any], *, path: UPath) -> list[float]:
    for transform in dataset.get("coordinateTransformations", ()):
        if transform.get("type") == "scale":
            return transform["scale"]
    raise CorruptImageError(
        f"OME-Zarr dataset {dataset.get('path')!r} of {path} has no 'scale' "
        "coordinateTransformation.",
        path=path,
    )


def _resolution_key(
    dataset: dict[str, Any], axes: list[dict[str, Any]], *, path: UPath
) -> float:
    """Product of the spatial-axis scale factors — smaller is finer. Uses the
    group's `axes` metadata to know which scale entries are spatial, rather
    than assuming any particular list order."""
    scale = _scale_transform(dataset, path=path)
    if axes:
        spatial_indices = [
            i for i, axis in enumerate(axes) if axis.get("type") == "space"
        ]
    else:
        spatial_indices = list(range(len(scale)))
    product = 1.0
    for i in spatial_indices:
        product *= scale[i]
    return product


def _ome_axis_name(axis: dict[str, Any], *, path: UPath) -> str:
    """The name (`t`/`c`/`z`/`y`/`x`) of one OME `axes` entry, from its `type`
    (channel/time/space) and, for a space axis, its `name`."""
    axis_type = axis.get("type")
    if axis_type == "channel":
        return "c"
    if axis_type == "time":
        return "t"
    if axis_type in ("space", None):
        return normalize_axis(axis.get("name", ""), path=path)
    raise UnsupportedImageDataError(
        f"Cannot place OME axis {axis!r} of {path} — only channel/time/space "
        "axes are supported.",
        path=path,
    )


@dataclass(frozen=True)
class OmeMultiscale:
    """The first `multiscales` entry of an OME-Zarr group, resolved to
    something a caller can pick a level from."""

    dataset_paths: tuple[str, ...]
    """Each dataset's own `path` (relative to the group), ranked finest
    first — rank 0 is `dataset_paths[0]`, regardless of the metadata's own
    dataset order."""

    axis_names: tuple[str, ...] | None
    """Axis names (`t`/`c`/`z`/`y`/`x`) derived from the group's `axes`
    metadata, in physical dimension order. None when the group has no `axes`
    entries (Zarr v2 permits this)."""


def resolve_ome_multiscale(attributes: dict[str, Any], *, path: UPath) -> OmeMultiscale:
    """
    Validates and ranks the first `multiscales` entry of an OME-Zarr group's
    attributes. `attributes` is the group's `.zattrs` (NGFF 0.4) or its
    `attributes.ome` (NGFF 0.5) — whichever carries `multiscales` and,
    for 0.4, `version` alongside it.
    """
    multiscales = attributes.get("multiscales")
    if not multiscales:
        raise UnsupportedImageFormatError(
            f"{path} is a Zarr group but not an OME-Zarr multiscale group "
            "(no 'multiscales' metadata).",
            path=path,
        )
    multiscale = multiscales[0]

    # NGFF 0.4 (Zarr v2) carries "version" inside the multiscale entry; NGFF
    # 0.5 (Zarr v3) carries it on the enclosing "ome" object instead —
    # `attributes` is whichever of the two was passed in.
    version = attributes.get("version") or multiscale.get("version")
    if version not in _SUPPORTED_OME_VERSIONS:
        raise UnsupportedImageFormatError(
            f"{path} is an OME-Zarr multiscale group with version "
            f"{version!r}; only {', '.join(_SUPPORTED_OME_VERSIONS)} are "
            "supported.",
            path=path,
        )

    datasets = multiscale.get("datasets") or []
    if not datasets:
        raise CorruptImageError(
            f"OME-Zarr multiscale group {path} has no datasets.", path=path
        )
    axes = multiscale.get("axes", [])

    ranked_indices = sorted(
        range(len(datasets)),
        key=lambda i: _resolution_key(datasets[i], axes, path=path),
    )
    dataset_paths = tuple(datasets[i]["path"] for i in ranked_indices)
    axis_names = (
        tuple(_ome_axis_name(axis, path=path) for axis in axes) if axes else None
    )
    return OmeMultiscale(dataset_paths, axis_names)

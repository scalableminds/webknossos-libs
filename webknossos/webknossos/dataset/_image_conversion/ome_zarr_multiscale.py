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

from ...dataset_properties import LayerViewConfiguration
from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from ..layer.abstract_layer import _UNALLOWED_LAYER_NAME_CHARS
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
class OmeChannelMetadata:
    """One entry of `omero.channels`, resolved to what a caller can apply
    directly to a layer."""

    view_configuration: LayerViewConfiguration | None
    """Color, intensity range, min/max and is_disabled, built from `color`/
    `window`/`active`. None if the entry carried none of these."""

    label: str | None
    """The channel's `label`, if any non-empty string was given."""


def _hex_to_rgb(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return None


def _parse_omero_channel(channel: dict[str, Any]) -> OmeChannelMetadata:
    color = _hex_to_rgb(channel.get("color"))
    window = channel.get("window")
    window = window if isinstance(window, dict) else {}
    start, end = window.get("start"), window.get("end")
    intensity_range = (start, end) if start is not None and end is not None else None
    is_disabled = not channel["active"] if "active" in channel else None

    view_configuration = None
    if any(
        v is not None
        for v in (
            color,
            intensity_range,
            window.get("min"),
            window.get("max"),
            is_disabled,
        )
    ):
        view_configuration = LayerViewConfiguration(
            color=color,
            intensity_range=intensity_range,
            min=window.get("min"),
            max=window.get("max"),
            is_disabled=is_disabled,
        )

    label = channel.get("label")
    label = label if isinstance(label, str) and label.strip() else None
    return OmeChannelMetadata(view_configuration, label)


def _parse_omero_channels(
    attributes: dict[str, Any],
) -> tuple[OmeChannelMetadata, ...] | None:
    omero = attributes.get("omero")
    channels = omero.get("channels") if isinstance(omero, dict) else None
    if not isinstance(channels, list) or not channels:
        return None
    return tuple(
        _parse_omero_channel(channel)
        if isinstance(channel, dict)
        else OmeChannelMetadata(None, None)
        for channel in channels
    )


def _omero_channel_at(
    channels: tuple[OmeChannelMetadata, ...] | None,
    channel_index: int | None,
    num_channels: int,
) -> OmeChannelMetadata | None:
    if channels is None:
        return None
    if channel_index is None:
        channel_index = 0 if num_channels == 1 else None
    if channel_index is None or not (0 <= channel_index < len(channels)):
        return None
    return channels[channel_index]


def suggested_view_configuration(
    channels: tuple[OmeChannelMetadata, ...] | None,
    channel_index: int | None,
    num_channels: int,
) -> LayerViewConfiguration | None:
    """The `omero`-derived view configuration for the channel this source
    writes: `channel_index` if pinned, else channel 0 when there is only a
    single output channel. None if there's no matching `omero` entry."""
    channel = _omero_channel_at(channels, channel_index, num_channels)
    return channel.view_configuration if channel else None


def layer_split_label(
    channels: tuple[OmeChannelMetadata, ...] | None, key: str, value: int
) -> str | None:
    """A layer-name suffix component for one `get_layer_split_options()` split
    entry, from the channel's `omero` label. Only resolves anything for
    `key == "channel"`; None when there's no usable label, so the caller
    falls back to its own default naming."""
    if key != "channel" or channels is None or not (0 <= value < len(channels)):
        return None
    channel = channels[value]
    if channel.label is None:
        return None
    sanitized = _UNALLOWED_LAYER_NAME_CHARS.sub("", channel.label).lstrip(".")
    return sanitized or None


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

    channels: tuple[OmeChannelMetadata, ...] | None
    """Per-channel `omero.channels` metadata, index-aligned with the group's
    channel axis. None when the group has no `omero` metadata."""


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
    channels = _parse_omero_channels(attributes)
    return OmeMultiscale(dataset_paths, axis_names, channels)

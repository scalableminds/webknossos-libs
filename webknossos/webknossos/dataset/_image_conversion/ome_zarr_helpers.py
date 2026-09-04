"""Shared OME-Zarr multiscale group parsing: ranking a group's `datasets` by
resolution, naming its axes, and reading their coordinate transformations. Used
by both `ZarrImageSource` (a multiscale group stored as a directory) and
`OzxImageSource` (one zipped into an `.ozx` archive) — the `multiscales`
metadata itself is identical either way, only how the resolved dataset path is
turned into something openable differs.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from upath import UPath

from ...dataset_properties import (
    _LENGTH_UNIT_TO_NANOMETER,
    AffineCoordinateTransformation,
    LayerViewConfiguration,
    LengthUnit,
    VoxelSize,
    length_unit_from_str,
)
from ...geometry.constants import C_AXIS, T_AXIS, XYZ_AXES
from ...geometry.vec3_float import Vec3Float
from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from ..layer.abstract_layer import _UNALLOWED_LAYER_NAME_CHARS
from .tensorstore_chunked_image_source import normalize_axis

_SUPPORTED_OME_VERSIONS = ("0.4", "0.5", "0.6rc1", "0.6")
_OME_06_VERSIONS = ("0.6rc1", "0.6")
_MAX_SEQUENCE_DEPTH = 8


@dataclass(frozen=True)
class OmeTransform:
    """One dataset's coordinate transformation, reduced to a scale and a
    translation. Both are index-aligned with the multiscale's axes, not with
    x/y/z — that mapping happens in `suggested_voxel_size` and
    `suggested_coordinate_transformations`."""

    scale: tuple[float, ...]
    translation: tuple[float, ...]

    @classmethod
    def identity(cls, ndim: int) -> OmeTransform:
        return cls((1.0,) * ndim, (0.0,) * ndim)

    def is_identity(self) -> bool:
        return all(s == 1.0 for s in self.scale) and all(
            t == 0.0 for t in self.translation
        )


def _compose(outer: OmeTransform, inner: OmeTransform) -> OmeTransform:
    """`outer` applied after `inner`."""
    return OmeTransform(
        tuple(o * i for o, i in zip(outer.scale, inner.scale)),
        tuple(
            o * i + t
            for o, i, t in zip(outer.scale, inner.translation, outer.translation)
        ),
    )


def _scale_transform(dataset: dict[str, Any], *, path: UPath) -> list[float]:
    for transform in dataset.get("coordinateTransformations", ()):
        if transform.get("type") == "scale":
            return transform["scale"]
    raise CorruptImageError(
        f"OME-Zarr dataset {dataset.get('path')!r} of {path} has no 'scale' "
        "coordinateTransformation.",
        path=path,
    )


def _translation_transform(dataset: dict[str, Any]) -> list[float] | None:
    for transform in dataset.get("coordinateTransformations", ()):
        if transform.get("type") == "translation":
            return transform["translation"]
    return None


def _legacy_transform(dataset: dict[str, Any], *, path: UPath) -> OmeTransform:
    """The scale and translation of one NGFF 0.4/0.5 dataset entry, which lists
    them side by side rather than nesting them in a `sequence`."""
    scale = _scale_transform(dataset, path=path)
    translation = _translation_transform(dataset) or [0.0] * len(scale)
    return OmeTransform(tuple(scale), tuple(translation))


def _literal_values(
    transform: dict[str, Any], key: str, *, ndim: int, path: UPath
) -> tuple[float, ...]:
    values = transform.get(key)
    if not isinstance(values, list):
        raise UnsupportedImageDataError(
            f"The OME-Zarr {transform.get('type')!r} coordinateTransformation of "
            f"{path} does not give {key!r} as a list of numbers — one stored in "
            "an array is not supported.",
            path=path,
        )
    if len(values) != ndim:
        raise CorruptImageError(
            f"The OME-Zarr {transform.get('type')!r} coordinateTransformation of "
            f"{path} has {len(values)} {key} entries, but the group has {ndim} axes.",
            path=path,
        )
    return tuple(float(value) for value in values)


def _flatten_transform(
    transform: dict[str, Any], *, ndim: int, path: UPath, depth: int = 0
) -> OmeTransform:
    """One NGFF 0.6 coordinateTransformation reduced to a scale and a
    translation. A `sequence` is composed entry by entry."""
    transform_type = transform.get("type")
    if transform_type == "identity":
        return OmeTransform.identity(ndim)
    if transform_type == "scale":
        return OmeTransform(
            _literal_values(transform, "scale", ndim=ndim, path=path), (0.0,) * ndim
        )
    if transform_type == "translation":
        return OmeTransform(
            (1.0,) * ndim,
            _literal_values(transform, "translation", ndim=ndim, path=path),
        )
    if transform_type == "sequence":
        if depth >= _MAX_SEQUENCE_DEPTH:
            raise UnsupportedImageDataError(
                f"The OME-Zarr coordinateTransformation of {path} nests "
                "'sequence' too deeply.",
                path=path,
            )
        result = OmeTransform.identity(ndim)
        for inner in transform.get("transformations") or ():
            result = _compose(
                _flatten_transform(inner, ndim=ndim, path=path, depth=depth + 1),
                result,
            )
        return result
    raise UnsupportedImageDataError(
        f"The OME-Zarr coordinateTransformation of {path} is of type "
        f"{transform_type!r}; only identity, scale, translation and sequences "
        "of them are supported.",
        path=path,
    )


def _ome_06_transform(
    dataset: dict[str, Any], *, ndim: int, path: UPath
) -> OmeTransform:
    transforms = dataset.get("coordinateTransformations") or []
    if len(transforms) != 1:
        raise CorruptImageError(
            f"OME-Zarr dataset {dataset.get('path')!r} of {path} has "
            f"{len(transforms)} coordinateTransformations; NGFF 0.6 requires "
            "exactly one.",
            path=path,
        )
    return _flatten_transform(transforms[0], ndim=ndim, path=path)


def _ome_06_axes(
    multiscale: dict[str, Any], datasets: list[dict[str, Any]], *, path: UPath
) -> list[dict[str, Any]]:
    """The axes of the intrinsic coordinate system — the one the first dataset's
    transformation names as its `output`, or the first system otherwise. NGFF
    0.6 moved `axes` from the multiscale entry into `coordinateSystems`."""
    systems = multiscale.get("coordinateSystems") or []
    if not systems:
        raise CorruptImageError(
            f"OME-Zarr multiscale group {path} has no coordinateSystems.", path=path
        )
    intrinsic_name = None
    transforms = datasets[0].get("coordinateTransformations") or []
    if transforms:
        output = transforms[0].get("output")
        if isinstance(output, dict):
            intrinsic_name = output.get("name")
    system = next(
        (s for s in systems if s.get("name") == intrinsic_name),
        systems[0],
    )
    axes = system.get("axes") or []
    if not axes:
        raise CorruptImageError(
            f"The OME-Zarr coordinate system {system.get('name')!r} of {path} "
            "has no axes.",
            path=path,
        )
    return axes


def _ome_top_level_transform(
    multiscale: dict[str, Any], *, ndim: int, path: UPath
) -> OmeTransform | None:
    """`multiscales[0].coordinateTransformations`, which places the whole
    pyramid in another coordinate system. None when absent, the identity, or of
    a type we cannot represent — it is optional metadata, so an unsupported one
    is warned about rather than raised."""
    transforms = multiscale.get("coordinateTransformations") or []
    if not transforms:
        return None
    result = OmeTransform.identity(ndim)
    try:
        for transform in transforms:
            result = _compose(
                _flatten_transform(transform, ndim=ndim, path=path), result
            )
    except (UnsupportedImageDataError, CorruptImageError):
        warnings.warn(
            f"[INFO] Ignoring the top-level coordinateTransformation of {path}, "
            "which is not supported."
        )
        return None
    return None if result.is_identity() else result


def _resolution_key(scale: tuple[float, ...], axes: list[dict[str, Any]]) -> float:
    """Product of the spatial-axis scale factors — smaller is finer. Uses the
    group's `axes` metadata to know which scale entries are spatial, rather
    than assuming any particular list order."""
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
        return C_AXIS
    if axis_type == "time":
        return T_AXIS
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
    """The `omero`-derived view configuration for the channel:
    `channel_index` if pinned, else channel 0 when there is only a
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

    transforms: tuple[OmeTransform, ...] = ()
    """Each dataset's coordinate transformation, index-aligned with
    `dataset_paths` and therefore ranked finest first. Each one's components
    are index-aligned with `axis_names`."""

    axis_units: tuple[str | None, ...] | None = None
    """Each axis' `unit` string, index-aligned with `axis_names`. None when the
    group has no `axes` entries."""

    top_level_transform: OmeTransform | None = None
    """The multiscale entry's own `coordinateTransformations`, placing the whole
    pyramid in another coordinate system. None when there is none."""


def resolve_ome_multiscale(attributes: dict[str, Any], *, path: UPath) -> OmeMultiscale:
    """
    Validates and ranks the first `multiscales` entry of an OME-Zarr group's
    attributes. `attributes` is the group's `.zattrs` (NGFF 0.4) or its
    `attributes.ome` (NGFF 0.5 and 0.6) — whichever carries `multiscales` and,
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
    # 0.5 and 0.6 (Zarr v3) carry it on the enclosing "ome" object instead —
    # `attributes` is whichever of the two was passed in.
    version_from_attributes = attributes.get("version")
    version = version_from_attributes or multiscale.get("version")
    if version not in _SUPPORTED_OME_VERSIONS:
        raise UnsupportedImageFormatError(
            f"{path} is an OME-Zarr multiscale group with version "
            f"{version!r}; only {', '.join(_SUPPORTED_OME_VERSIONS)} are "
            "supported.",
            path=path,
        )
    is_06 = version in _OME_06_VERSIONS
    if is_06 and version_from_attributes is None:
        raise UnsupportedImageFormatError(
            f"{path} declares OME-Zarr version {version!r} inside its multiscale "
            "entry; NGFF 0.6 requires it on the enclosing 'ome' object.",
            path=path,
        )

    datasets = multiscale.get("datasets") or []
    if not datasets:
        raise CorruptImageError(
            f"OME-Zarr multiscale group {path} has no datasets.", path=path
        )
    axes = (
        _ome_06_axes(multiscale, datasets, path=path)
        if is_06
        else multiscale.get("axes", [])
    )
    ndim = len(axes) or len(_scale_transform(datasets[0], path=path))
    transforms = [
        _ome_06_transform(dataset, ndim=ndim, path=path)
        if is_06
        else _legacy_transform(dataset, path=path)
        for dataset in datasets
    ]
    top_level_transform = _ome_top_level_transform(multiscale, ndim=ndim, path=path)

    ranked_indices = sorted(
        range(len(datasets)),
        key=lambda i: _resolution_key(transforms[i].scale, axes),
    )
    dataset_paths = tuple(datasets[i]["path"] for i in ranked_indices)
    axis_names = (
        tuple(_ome_axis_name(axis, path=path) for axis in axes) if axes else None
    )
    axis_units = tuple(axis.get("unit") for axis in axes) if axes else None
    channels = _parse_omero_channels(attributes)
    return OmeMultiscale(
        dataset_paths=dataset_paths,
        axis_names=axis_names,
        channels=channels,
        transforms=tuple(transforms[i] for i in ranked_indices),
        axis_units=axis_units,
        top_level_transform=top_level_transform,
    )


def _spatial_components(
    multiscale: OmeMultiscale, rank: int
) -> tuple[OmeTransform, dict[str, int]] | None:
    """The transform of the level at `rank` plus where each of x/y/z sits in
    it. None when the group's axes or that rank are unusable."""
    if multiscale.axis_names is None or not (0 <= rank < len(multiscale.transforms)):
        return None
    index_of = {axis: i for i, axis in enumerate(multiscale.axis_names)}
    return multiscale.transforms[rank], index_of


def suggested_voxel_size(multiscale: OmeMultiscale, rank: int) -> VoxelSize | None:
    """The physical size of one voxel of the level at `rank`, from its `scale`
    coordinateTransformation and the axes' `unit`. `rank` is the level that
    actually gets opened, since that one becomes the layer's Mag(1).

    None when the metadata does not determine a voxel size — no axes, no unit
    on any spatial axis, or an unusable factor. Guessing a unit would fabricate
    a physical scale, so the caller is left to supply one instead.
    """
    components = _spatial_components(multiscale, rank)
    if components is None or multiscale.axis_units is None:
        return None
    transform, index_of = components
    top = multiscale.top_level_transform

    factors: list[float] = []
    units: list[LengthUnit | None] = []
    for axis in XYZ_AXES:
        i = index_of.get(axis)
        if i is None:
            # An axis the data does not have, e.g. z for a 2D image.
            factors.append(1.0)
            units.append(None)
            continue
        factors.append(transform.scale[i] * (top.scale[i] if top else 1.0))
        unit = multiscale.axis_units[i]
        if unit is None:
            units.append(None)
            continue
        try:
            units.append(length_unit_from_str(unit))
        except ValueError:
            warnings.warn(
                f"[INFO] Ignoring the OME-Zarr voxel size, because the axis unit "
                f"{unit!r} is not a known length unit."
            )
            return None

    present_units = [unit for unit in units if unit is not None]
    if not present_units:
        return None
    common = min(present_units, key=lambda unit: _LENGTH_UNIT_TO_NANOMETER[unit])
    factors = [
        factor
        * (
            _LENGTH_UNIT_TO_NANOMETER[unit] / _LENGTH_UNIT_TO_NANOMETER[common]
            if unit is not None
            else 1.0
        )
        for factor, unit in zip(factors, units)
    ]
    if not all(math.isfinite(factor) and factor > 0 for factor in factors):
        return None
    return VoxelSize(Vec3Float(*factors), common)


def suggested_coordinate_transformations(
    multiscale: OmeMultiscale, rank: int
) -> tuple[AffineCoordinateTransformation, ...] | None:
    """Where the level at `rank` sits relative to the origin, as a layer
    coordinate transformation in that level's own voxels.

    The scale already went into `suggested_voxel_size`, so what is left is the
    translation, divided by the scale to express it in voxels. None when the
    level sits at the origin.
    """
    components = _spatial_components(multiscale, rank)
    if components is None:
        return None
    transform, index_of = components
    top = multiscale.top_level_transform

    translation: list[float] = []
    for axis in XYZ_AXES:
        i = index_of.get(axis)
        if i is None:
            translation.append(0.0)
            continue
        top_scale = top.scale[i] if top else 1.0
        top_translation = top.translation[i] if top else 0.0
        scale = transform.scale[i] * top_scale
        translation.append(
            (top_scale * transform.translation[i] + top_translation) / scale
            if scale
            else 0.0
        )

    if not all(math.isfinite(value) for value in translation):
        return None
    if all(value == 0.0 for value in translation):
        return None
    return (AffineCoordinateTransformation.from_translation(translation),)

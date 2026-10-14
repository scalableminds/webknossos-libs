"""Shared OME-Zarr multiscale group parsing: ranking a group's `datasets` by
resolution and naming its axes. Used by both `ZarrImageSource` (a multiscale
group stored as a directory) and `OzxImageSource` (one zipped into an `.ozx`
archive) — the `multiscales` metadata itself is identical either way, only
how the resolved dataset path is turned into something openable differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from upath import UPath

from ...dataset_properties import AffineCoordinateTransformation, LayerViewConfiguration
from ...geometry.constants import C_AXIS, T_AXIS
from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from ..layer.abstract_layer import _UNALLOWED_LAYER_NAME_CHARS
from .tensorstore_chunked_image_source import normalize_axis

_SUPPORTED_OME_VERSIONS = ("0.4", "0.5", "0.6")


def _scale_transform(dataset: dict[str, Any], *, path: UPath) -> list[float]:
    for transform in dataset.get("coordinateTransformations", ()):
        if transform.get("type") == "scale":
            return transform["scale"]
    raise CorruptImageError(
        f"OME-Zarr dataset {dataset.get('path')!r} of {path} has no 'scale' "
        "coordinateTransformation.",
        path=path,
    )


def _single_dataset_transform(
    dataset: dict[str, Any], *, path: UPath
) -> dict[str, Any]:
    transforms = dataset.get("coordinateTransformations")
    if not isinstance(transforms, list) or len(transforms) != 1:
        raise CorruptImageError(
            f"OME-Zarr dataset {dataset.get('path')!r} of {path} must have "
            "exactly one coordinateTransformation.",
            path=path,
        )
    return transforms[0]


def _transform_parameters(
    transform: dict[str, Any], key: str, shape: tuple[int, ...], *, path: UPath
) -> np.ndarray:
    if key not in transform and "path" in transform:
        raise UnsupportedImageDataError(
            f"The {transform['type']!r} coordinateTransformation of {path} stores "
            "its parameters in a Zarr array, which is not supported.",
            path=path,
        )
    try:
        parameters = np.array(transform[key], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        parameters = None
    if parameters is None or parameters.shape != shape:
        raise CorruptImageError(
            f"The {transform.get('type')!r} coordinateTransformation of {path} "
            f"must have a {key!r} of shape {shape}.",
            path=path,
        )
    return parameters


def _transform_matrix(
    transform: dict[str, Any], ndim: int, *, path: UPath
) -> np.ndarray:
    """A coordinateTransformation as an (ndim + 1) x (ndim + 1) homogeneous
    matrix, mapping array indices to the intrinsic coordinate system."""
    transform_type = transform.get("type")
    matrix = np.eye(ndim + 1)
    if transform_type == "identity":
        pass
    elif transform_type == "scale":
        scale = _transform_parameters(transform, "scale", (ndim,), path=path)
        matrix[:ndim, :ndim] = np.diag(scale)
    elif transform_type == "translation":
        translation = _transform_parameters(
            transform, "translation", (ndim,), path=path
        )
        matrix[:ndim, ndim] = translation
    elif transform_type == "rotation":
        matrix[:ndim, :ndim] = _transform_parameters(
            transform, "rotation", (ndim, ndim), path=path
        )
    elif transform_type == "affine":
        matrix[:ndim, :] = _transform_parameters(
            transform, "affine", (ndim, ndim + 1), path=path
        )
    elif transform_type == "mapAxis":
        order = _transform_parameters(transform, "mapAxis", (ndim,), path=path)
        if sorted(order) != list(range(ndim)):
            raise CorruptImageError(
                f"The 'mapAxis' coordinateTransformation of {path} must be a "
                f"permutation of 0..{ndim - 1}, got {order.tolist()}.",
                path=path,
            )
        matrix[:ndim, :ndim] = np.eye(ndim)[order.astype(int)]
    elif transform_type == "sequence":
        # The first transformation is applied first.
        for inner in transform.get("transformations") or []:
            matrix = _transform_matrix(inner, ndim, path=path) @ matrix
    else:
        raise UnsupportedImageDataError(
            f"The coordinateTransformation {transform_type!r} of {path} is not "
            "supported; only 'identity', 'scale', 'translation', 'rotation', "
            "'affine', 'mapAxis' and 'sequence' are.",
            path=path,
        )
    return matrix


def _spatial_matrix(
    matrix: np.ndarray, axes: list[dict[str, Any]], *, path: UPath
) -> np.ndarray:
    """The x/y/z part of a dataset's transform as a 4x4 matrix in x, y, z
    order. Axes missing from the image (e.g. z for 2D) stay untouched."""
    ndim = len(axes)
    spatial = [i for i, axis in enumerate(axes) if axis.get("type") == "space"]
    other = [i for i in range(ndim) if i not in spatial]
    if np.any(matrix[np.ix_(spatial, other)]) or np.any(matrix[np.ix_(other, spatial)]):
        raise UnsupportedImageDataError(
            f"A coordinateTransformation of {path} mixes spatial axes with "
            "non-spatial (e.g. channel or time) axes, which is not supported.",
            path=path,
        )
    xyz_index = {"x": 0, "y": 1, "z": 2}
    rows = [xyz_index[_ome_axis_name(axes[i], path=path)] for i in spatial]
    result = np.eye(4)
    for row, i in zip(rows, spatial):
        for column, j in zip(rows, spatial):
            result[row, column] = matrix[i, j]
        result[row, 3] = matrix[i, ndim]
    return result


def _layer_transformation(
    spatial_matrix: np.ndarray, *, path: UPath
) -> AffineCoordinateTransformation | None:
    """The WEBKNOSSOS layer transformation for a dataset's x/y/z transform,
    in voxel units of that dataset. None if it is a plain scale."""
    voxel_size = np.linalg.norm(spatial_matrix[:3, :3], axis=0)
    if not np.all(voxel_size > 0):
        raise CorruptImageError(
            f"A coordinateTransformation of {path} collapses a spatial axis.",
            path=path,
        )
    in_voxels = np.diag([*(1 / voxel_size), 1.0]) @ spatial_matrix
    if np.allclose(in_voxels, np.eye(4)):
        return None
    return AffineCoordinateTransformation(matrix=in_voxels)


def _layer_transformation_0_4(
    dataset: dict[str, Any], axes: list[dict[str, Any]], *, path: UPath
) -> AffineCoordinateTransformation | None:
    """The WEBKNOSSOS layer transformation for a dataset's `scale` and
    optional `translation`. None if there is no translation, or no `axes`
    to tell which of its entries are spatial."""
    translations = [
        transform
        for transform in dataset.get("coordinateTransformations", ())
        if transform.get("type") == "translation"
    ]
    if not translations or not axes:
        return None
    scale = {"type": "scale", "scale": _scale_transform(dataset, path=path)}
    matrix = _transform_matrix(
        {"type": "sequence", "transformations": [scale, translations[0]]},
        len(axes),
        path=path,
    )
    return _layer_transformation(_spatial_matrix(matrix, axes, path=path), path=path)


def _intrinsic_axes_0_6(
    multiscale: dict[str, Any], datasets: list[dict[str, Any]], *, path: UPath
) -> list[dict[str, Any]]:
    """The `axes` of the coordinate system all datasets map into."""
    output_names = {
        (_single_dataset_transform(dataset, path=path).get("output") or {}).get("name")
        for dataset in datasets
    }
    if len(output_names) != 1:
        raise CorruptImageError(
            f"The datasets of OME-Zarr multiscale group {path} must all map "
            f"into the same coordinate system, got outputs {output_names!r}.",
            path=path,
        )
    (name,) = output_names
    for coordinate_system in multiscale.get("coordinateSystems") or []:
        if coordinate_system.get("name") == name:
            return coordinate_system.get("axes") or []
    raise CorruptImageError(
        f"OME-Zarr multiscale group {path} has no coordinate system named {name!r}.",
        path=path,
    )


def _resolution_key(scale: list[float], axes: list[dict[str, Any]]) -> float:
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

    layer_transformations: tuple[AffineCoordinateTransformation | None, ...]
    """Per dataset (ranked like `dataset_paths`), the WEBKNOSSOS layer
    transformation its coordinateTransformations describe beyond a plain
    scale. None where there is none."""


def resolve_ome_multiscale(attributes: dict[str, Any], *, path: UPath) -> OmeMultiscale:
    """
    Validates and ranks the first `multiscales` entry of an OME-Zarr group's
    attributes. `attributes` is the group's `.zattrs` (NGFF 0.4) or its
    `attributes.ome` (NGFF 0.5/0.6) — whichever carries `multiscales` and,
    for 0.4, `version` alongside it.
    """
    multiscales = attributes.get("multiscales")
    if not multiscales and "scene" in attributes:
        raise UnsupportedImageFormatError(
            f"{path} is an OME-Zarr scene group. Convert the multiscale "
            "images it contains individually instead.",
            path=path,
        )
    if not multiscales:
        raise UnsupportedImageFormatError(
            f"{path} is a Zarr group but not an OME-Zarr multiscale group "
            "(no 'multiscales' metadata).",
            path=path,
        )
    multiscale = multiscales[0]

    # NGFF 0.4 (Zarr v2) carries "version" inside the multiscale entry; NGFF
    # 0.5/0.6 (Zarr v3) carry it on the enclosing "ome" object instead —
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
    layer_transformations: list[AffineCoordinateTransformation | None]
    if version == "0.6":
        # NGFF 0.6 moves "axes" into named "coordinateSystems", and a
        # dataset's transform can be any composition of affine transforms.
        axes = _intrinsic_axes_0_6(multiscale, datasets, path=path)
        spatial_matrices = [
            _spatial_matrix(
                _transform_matrix(
                    _single_dataset_transform(dataset, path=path),
                    len(axes),
                    path=path,
                ),
                axes,
                path=path,
            )
            for dataset in datasets
        ]
        # The volume of a voxel — smaller is finer.
        resolution_keys = [abs(np.linalg.det(m[:3, :3])) for m in spatial_matrices]
        layer_transformations = [
            _layer_transformation(m, path=path) for m in spatial_matrices
        ]
    else:
        axes = multiscale.get("axes", [])
        resolution_keys = [
            _resolution_key(_scale_transform(dataset, path=path), axes)
            for dataset in datasets
        ]
        layer_transformations = [
            _layer_transformation_0_4(dataset, axes, path=path) for dataset in datasets
        ]

    ranked_indices = sorted(range(len(datasets)), key=lambda i: resolution_keys[i])
    dataset_paths = tuple(datasets[i]["path"] for i in ranked_indices)
    axis_names = (
        tuple(_ome_axis_name(axis, path=path) for axis in axes) if axes else None
    )
    channels = _parse_omero_channels(attributes)
    return OmeMultiscale(
        dataset_paths,
        axis_names,
        channels,
        tuple(layer_transformations[i] for i in ranked_indices),
    )

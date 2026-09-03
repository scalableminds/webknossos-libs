import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ..dataset_properties import DataFormat
from ..geometry.constants import C_AXIS, T_AXIS, X_AXIS, Y_AXIS, Z_AXIS
from ..utils import is_writable_path
from .defaults import ZARR_JSON_FILE_NAME, ZATTRS_FILE_NAME, ZGROUP_FILE_NAME

if TYPE_CHECKING:
    from .dataset import Dataset
    from .layer import Layer, MagView

_AXIS_TYPES = {
    C_AXIS: "channel",
    T_AXIS: "time",
    X_AXIS: "space",
    Y_AXIS: "space",
    Z_AXIS: "space",
}
_SPACE_AXES = (X_AXIS, Y_AXIS, Z_AXIS)


def _ome_axes(axes_order: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for axis in axes_order:
        entry: dict[str, Any] = {"name": axis}
        axis_type = _AXIS_TYPES.get(axis)
        if axis_type is not None:
            entry["type"] = axis_type
            if axis in _SPACE_AXES:
                entry["unit"] = "nanometer"
        result.append(entry)
    return result


def _ome_scale(
    axes_order: tuple[str, ...], dataset: "Dataset", mag: "MagView"
) -> list[float]:
    voxel_size = np.array(dataset.voxel_size)
    mag_np = mag.mag.to_np()
    return [
        float(voxel_size[_SPACE_AXES.index(axis)] * mag_np[_SPACE_AXES.index(axis)])
        if axis in _SPACE_AXES
        else 1.0
        for axis in axes_order
    ]


def get_ome_0_5_multiscale_metadata(
    dataset: "Dataset", layer: "Layer"
) -> dict[str, Any]:
    axes_order = layer.normalized_bounding_box.axes
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": _ome_axes(axes_order),
                    "datasets": [
                        {
                            "path": mag.path.name,
                            "coordinateTransformations": [
                                {
                                    "type": "scale",
                                    "scale": _ome_scale(axes_order, dataset, mag),
                                }
                            ],
                        }
                        for mag in layer.mags.values()
                    ],
                }
            ],
        }
    }


def get_ome_0_4_multiscale_metadata(
    dataset: "Dataset", layer: "Layer"
) -> dict[str, Any]:
    axes_order = layer.normalized_bounding_box.axes
    return {
        "multiscales": [
            {
                "version": "0.4",
                "axes": _ome_axes(axes_order),
                "datasets": [
                    {
                        "path": mag.path.name,
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": _ome_scale(axes_order, dataset, mag),
                            }
                        ],
                    }
                    for mag in layer.mags.values()
                ],
            }
        ]
    }


def write_ome_metadata(dataset: "Dataset", layer: "Layer") -> None:
    if not is_writable_path(layer.path):
        return
    if layer.data_format == DataFormat.Zarr3:
        with (layer.path / ZARR_JSON_FILE_NAME).open("w", encoding="utf-8") as outfile:
            json.dump(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": get_ome_0_5_multiscale_metadata(dataset, layer),
                },
                outfile,
                indent=4,
            )
    if layer.data_format == DataFormat.Zarr:
        with (layer.path / ZGROUP_FILE_NAME).open("w", encoding="utf-8") as outfile:
            json.dump({"zarr_format": "2"}, outfile, indent=4)
        with (layer.path / ZATTRS_FILE_NAME).open("w", encoding="utf-8") as outfile:
            json.dump(
                get_ome_0_4_multiscale_metadata(dataset, layer),
                outfile,
                indent=4,
            )

import json
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from .data_format import DataFormat
from .defaults import ZARR_JSON_FILE_NAME, ZATTRS_FILE_NAME, ZGROUP_FILE_NAME

if TYPE_CHECKING:
    from .dataset import Dataset
    from .layer import Layer


def get_ome_0_4_multiscale_metadata(
    dataset: "Dataset", layer: "Layer"
) -> Dict[str, Any]:
    return {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {
                        "name": "x",
                        "type": "space",
                        "unit": "nanometer",
                    },
                    {
                        "name": "y",
                        "type": "space",
                        "unit": "nanometer",
                    },
                    {
                        "name": "z",
                        "type": "space",
                        "unit": "nanometer",
                    },
                ],
                "datasets": [
                    {
                        "path": mag.path.name,
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0]
                                + (
                                    np.array(dataset.voxel_size) * mag.mag.to_np()
                                ).tolist(),
                            }
                        ],
                    }
                    for mag in layer.mags.values()
                ],
            }
        ]
    }


def write_ome_0_4_metadata(dataset: "Dataset", layer: "Layer") -> None:
    if layer.data_format == DataFormat.Zarr3:
        with (layer.path / ZARR_JSON_FILE_NAME).open("w", encoding="utf-8") as outfile:
            json.dump(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": get_ome_0_4_multiscale_metadata(dataset, layer),
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

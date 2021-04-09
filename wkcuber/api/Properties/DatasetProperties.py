import json
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, cast

from wkcuber.api.Layer import Layer
from wkcuber.api.Properties.LayerProperties import (
    SegmentationLayerProperties,
    LayerProperties,
)
from wkcuber.api.Properties.ResolutionProperties import WkResolution, Resolution
from wkcuber.mag import Mag


class Properties:

    FILE_NAME = "datasource-properties.json"

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        scale: Tuple[float, float, float],
        team: str = "",
        data_layers: Dict[str, LayerProperties] = None,
    ) -> None:
        self._path = str(path)
        self._name = name
        self._team = team
        self._scale = scale
        if data_layers is None:
            self._data_layers = {}
        else:
            self._data_layers = data_layers

    @classmethod
    def _from_json(cls, path: Union[str, Path]) -> "Properties":
        pass

    def _export_as_json(self) -> None:
        pass

    def _add_layer(
        self,
        layer_name: str,
        category: str,
        element_class: str,
        data_format: str,
        num_channels: int = 1,
        **kwargs: Dict[str, Any]
    ) -> None:
        # this layer is already in data_layers in case we reconstruct the dataset from a datasource-properties.json
        if layer_name not in self.data_layers:
            if category == Layer.SEGMENTATION_TYPE:
                assert (
                    "largest_segment_id" in kwargs
                ), "When adding a segmentation layer, largest_segment_id has to be supplied."

                self.data_layers[layer_name] = SegmentationLayerProperties(
                    layer_name,
                    category,
                    element_class,
                    data_format,
                    num_channels,
                    largest_segment_id=cast(int, kwargs["largest_segment_id"]),
                )
            else:
                self.data_layers[layer_name] = LayerProperties(
                    layer_name, category, element_class, data_format, num_channels
                )
            self._export_as_json()

    def _add_mag(self, layer_name: str, mag: str, **kwargs: int) -> None:
        pass

    def _delete_layer(self, layer_name: str) -> None:
        del self.data_layers[layer_name]
        self._export_as_json()

    def _delete_mag(self, layer_name: str, mag: str) -> None:
        self._data_layers[layer_name]._delete_resolution(mag)
        self._export_as_json()

    def _set_bounding_box_of_layer(
        self, layer_name: str, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> None:
        self._data_layers[layer_name]._set_bounding_box_size(size)
        self._data_layers[layer_name]._set_bounding_box_offset(offset)
        self._export_as_json()

    def get_bounding_box_of_layer(
        self, layer_name: str
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        offset = self._data_layers[layer_name].get_bounding_box_offset()
        size = self._data_layers[layer_name].get_bounding_box_size()
        return offset, size

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def team(self) -> str:
        return self._team

    @property
    def scale(self) -> Tuple[float, float, float]:
        return self._scale

    @property
    def data_layers(self) -> dict:
        return self._data_layers


class WKProperties(Properties):
    @classmethod
    def _from_json(cls, path: Union[str, Path]) -> "WKProperties":
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers: Dict[str, LayerProperties] = {}
            for layer in data["dataLayers"]:
                if layer["category"] == Layer.SEGMENTATION_TYPE:
                    data_layers[layer["name"]] = SegmentationLayerProperties._from_json(
                        layer, WkResolution, path
                    )
                else:
                    data_layers[layer["name"]] = LayerProperties._from_json(
                        layer, WkResolution, path
                    )

            return cls(
                path, data["id"]["name"], data["scale"], data["id"]["team"], data_layers
            )

    def _export_as_json(self) -> None:
        data = {
            "id": {"name": self.name, "team": self.team},
            "scale": self.scale,
            "dataLayers": [
                self.data_layers[layer_name]._to_json()
                for layer_name in self.data_layers
            ],
        }
        with open(self.path, "w") as outfile:
            json.dump(data, outfile, indent=4, separators=(",", ": "))

    def _add_mag(self, layer_name: str, mag: str, **kwargs: int) -> None:
        assert "cube_length" in kwargs
        # this mag is already in wkw_magnifications in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_magnifications
            ]
        ):
            self._data_layers[layer_name]._add_resolution(
                WkResolution(mag, kwargs["cube_length"])
            )
            self._export_as_json()


class TiffProperties(Properties):
    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        scale: Tuple[float, float, float],
        pattern: str,
        team: str = "",
        data_layers: Dict[str, LayerProperties] = None,
        tile_size: Optional[Tuple[int, int]] = (32, 32),
    ) -> None:
        super().__init__(path, name, scale, team, data_layers)
        self.pattern = pattern
        self.tile_size = tile_size

    @classmethod
    def _from_json(cls, path: Union[str, Path]) -> Properties:
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers: Dict[str, LayerProperties] = {}
            for layer in data["dataLayers"]:
                if layer["category"] == Layer.SEGMENTATION_TYPE:
                    data_layers[layer["name"]] = SegmentationLayerProperties._from_json(
                        layer, Resolution, path
                    )
                else:
                    data_layers[layer["name"]] = LayerProperties._from_json(
                        layer, Resolution, path
                    )

            tile_size = data.get("tile_size")
            if tile_size is not None:
                tile_size = (tile_size[0], tile_size[1])
            return cls(
                path=path,
                name=data["id"]["name"],
                scale=data["scale"],
                pattern=data["pattern"],
                team=data["id"]["team"],
                data_layers=data_layers,
                tile_size=tile_size,
            )

    def _export_as_json(self) -> None:
        data = {
            "id": {"name": self.name, "team": self.team},
            "scale": self.scale,
            "pattern": self.pattern,
            "dataLayers": [
                self.data_layers[layer_name]._to_json()
                for layer_name in self.data_layers
            ],
        }
        if self.tile_size is not None:
            data["tile_size"] = self.tile_size

        with open(self.path, "w") as outfile:
            json.dump(data, outfile, indent=4, separators=(",", ": "))

    def _add_mag(self, layer_name: str, mag: str, **kwargs: int) -> None:
        # this mag is already in wkw_magnifications in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_magnifications
            ]
        ):
            self.data_layers[layer_name]._add_resolution(Resolution(mag))
            self._export_as_json()

import json

from wkcuber.api.Layer import Layer
from wkcuber.api.Properties.LayerProperties import (
    SegmentationLayerProperties,
    LayerProperties,
)
from wkcuber.api.Properties.ResolutionProperties import WkResolution, Resolution
from wkcuber.mag import Mag


class Properties:

    FILE_NAME = "datasource-properties.json"

    def __init__(self, path, name, scale, team="", data_layers=None):
        self._path = path
        self._name = name
        self._team = team
        self._scale = scale
        if data_layers is None:
            self._data_layers = {}
        else:
            self._data_layers = data_layers

    @classmethod
    def _from_json(cls, path):
        pass

    def _export_as_json(self):
        pass

    def _add_layer(
        self, layer_name, category, element_class, data_format, num_channels=1, **kwargs
    ):
        # this layer is already in data_layers in case we reconstruct the dataset from a datasource-properties.json
        if layer_name not in self.data_layers:
            if category == Layer.SEGMENTATION_TYPE:
                assert (
                    "largest_segment_id" in kwargs
                ), "When adding a segmentation layer, largest_segment_id has to be supplied."

                new_layer = SegmentationLayerProperties(
                    layer_name,
                    category,
                    element_class,
                    data_format,
                    num_channels,
                    largest_segment_id=kwargs["largest_segment_id"],
                )
            else:
                new_layer = LayerProperties(
                    layer_name, category, element_class, data_format, num_channels
                )
            self.data_layers[layer_name] = new_layer
            self._export_as_json()

    def _delete_layer(self, layer_name):
        del self.data_layers[layer_name]
        self._export_as_json()

    def _delete_mag(self, layer_name, mag):
        self._data_layers[layer_name]._delete_resolution(mag)
        self._export_as_json()

    def _set_bounding_box_of_layer(self, layer_name, offset, size):
        self._data_layers[layer_name]._set_bounding_box_size(size)
        self._data_layers[layer_name]._set_bounding_box_offset(offset)
        self._export_as_json()

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
    def scale(self) -> tuple:
        return self._scale

    @property
    def data_layers(self) -> dict:
        return self._data_layers


class WKProperties(Properties):
    @classmethod
    def _from_json(cls, path) -> Properties:
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
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

    def _export_as_json(self):
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

    def _add_mag(self, layer_name, mag, cube_length):
        # this mag is already in wkw_magnifications in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_magnifications
            ]
        ):
            self._data_layers[layer_name]._add_resolution(
                WkResolution(mag, cube_length)
            )
            self._export_as_json()


class TiffProperties(Properties):
    def __init__(
        self, path, name, scale, pattern, team="", data_layers=None, tile_size=(32, 32)
    ):
        super().__init__(path, name, scale, team, data_layers)
        self.pattern = pattern
        self.tile_size = tile_size

    @classmethod
    def _from_json(cls, path) -> Properties:
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
            for layer in data["dataLayers"]:
                if layer["category"] == Layer.SEGMENTATION_TYPE:
                    data_layers[layer["name"]] = SegmentationLayerProperties._from_json(
                        layer, Resolution, path
                    )
                else:
                    data_layers[layer["name"]] = LayerProperties._from_json(
                        layer, Resolution, path
                    )

            return cls(
                path=path,
                name=data["id"]["name"],
                scale=data["scale"],
                pattern=data["pattern"],
                team=data["id"]["team"],
                data_layers=data_layers,
                tile_size=data.get("tile_size"),
            )

    def _export_as_json(self):
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

    def _add_mag(self, layer_name, mag):
        # this mag is already in wkw_magnifications in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_magnifications
            ]
        ):
            self.data_layers[layer_name]._add_resolution(Resolution(mag))
            self._export_as_json()

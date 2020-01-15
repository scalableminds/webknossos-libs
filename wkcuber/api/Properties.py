import json
import numpy as np

from wkcuber.mag import Mag


class Resolution:
    def to_json(self):
        pass

    @classmethod
    def from_json(cls, json_data):
        pass


class TiffResolution(Resolution):
    def __init__(self, mag):
        self.mag = Mag(mag)

    def to_json(self):
        return {"resolution": self.mag.to_array()}

    @classmethod
    def from_json(cls, json_data):
        return cls(json_data["resolution"])


class WkResolution(Resolution):
    def __init__(self, mag, cube_length):
        self.mag = Mag(mag)
        self.cube_length = cube_length

    def to_json(self):
        return {"resolution": self.mag.to_array(), "cube_length": self.cube_length}

    @classmethod
    def from_json(cls, json_data):
        return cls(json_data["resolution"], json_data["cube_length"])


class Properties:
    def __init__(self, path, name, scale, team="", data_layers=None):
        self.path = path
        self.name = name
        self.team = team
        self.scale = scale
        if data_layers is None:
            self.data_layers = {}
        else:
            self.data_layers = data_layers

    @classmethod
    def from_json(cls, path):
        pass

    def export_as_json(self):
        pass

    def add_layer(self, layer_name, category, element_class, num_channels=1):
        pass

    def delete_layer(self, layer_name):
        del self.data_layers[layer_name]
        self.export_as_json()

    def delete_mag(self, layer_name, mag):
        self.data_layers[layer_name].wkw_resolutions = [
            r for r in self.data_layers[layer_name].wkw_resolutions if r.mag != Mag(mag)
        ]
        self.export_as_json()

    def set_bounding_box_size_of_layer(self, layer_name, size):
        self.data_layers[layer_name].set_bounding_box_size(size)
        self.export_as_json()

    def set_bounding_box_offset_of_layer(self, layer_name, offset):
        self.data_layers[layer_name].set_bounding_box_offset(offset)
        self.export_as_json()


class WKProperties(Properties):
    @classmethod
    def from_json(cls, path):
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
            for layer in data["dataLayers"]:
                data_layers[layer["name"]] = LayerProperties.from_json(
                    layer, WkResolution
                )

            return cls(
                path, data["id"]["name"], data["scale"], data["id"]["team"], data_layers
            )

    def export_as_json(self):
        data = {
            "id": {"name": self.name, "team": self.team},
            "scale": self.scale,
            "dataLayers": [
                self.data_layers[layer_name].to_json()
                for layer_name in self.data_layers
            ],
        }
        with open(self.path, "w") as outfile:
            json.dump(data, outfile, indent=4, separators=(",", ": "))

    def add_layer(self, layer_name, category, element_class, num_channels=1):
        # this layer is already in data_layers in case we reconstruct the dataset from a datasource-properties.json
        if layer_name not in self.data_layers:
            new_layer = LayerProperties(
                "wkw", layer_name, category, element_class, num_channels
            )
            self.data_layers[layer_name] = new_layer
            self.export_as_json()

    def add_mag(self, layer_name, mag, cube_length):
        # this mag is already in wkw_resolutions in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_resolutions
            ]
        ):
            self.data_layers[layer_name].add_resolution(WkResolution(mag, cube_length))
            self.export_as_json()


class TiffProperties(Properties):
    def __init__(
        self,
        path,
        name,
        scale,
        pattern,
        team="",
        data_layers=None,
        grid_shape=(0, 0),
        tile_size=(32, 32),
    ):
        super().__init__(path, name, scale, team, data_layers)
        self.grid_shape = grid_shape
        self.pattern = pattern
        self.tile_size = tile_size

    @classmethod
    def from_json(cls, path):
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
            for layer in data["dataLayers"]:
                data_layers[layer["name"]] = LayerProperties.from_json(
                    layer, TiffResolution
                )

            return cls(
                path=path,
                name=data["id"]["name"],
                scale=data["scale"],
                pattern=data["pattern"],
                team=data["id"]["team"],
                data_layers=data_layers,
                grid_shape=data["grid_shape"],
                tile_size=data.get("tile_size"),
            )

    def export_as_json(self):
        data = {
            "id": {"name": self.name, "team": self.team},
            "scale": self.scale,
            "pattern": self.pattern,
            "dataLayers": [
                self.data_layers[layer_name].to_json()
                for layer_name in self.data_layers
            ],
            "grid_shape": self.grid_shape,
            "tile_size": self.tile_size,
        }
        with open(self.path, "w") as outfile:
            json.dump(data, outfile, indent=4, separators=(",", ": "))

    def add_layer(self, layer_name, category, element_class="uint8", num_channels=1):
        # this layer is already in data_layers in case we reconstruct the dataset from a datasource-properties.json
        if layer_name not in self.data_layers:
            new_layer = LayerProperties(
                "tiff", layer_name, category, element_class, num_channels
            )
            self.data_layers[layer_name] = new_layer
            self.export_as_json()

    def add_mag(self, layer_name, mag):
        # this mag is already in wkw_resolutions in case we reconstruct the dataset from a datasource-properties.json
        if not any(
            [
                res.mag == Mag(mag)
                for res in self.data_layers[layer_name].wkw_resolutions
            ]
        ):
            self.data_layers[layer_name].add_resolution(TiffResolution(mag))
            self.export_as_json()


class LayerProperties:
    def __init__(
        self,
        data_format,
        name,
        category,
        element_class,
        num_channels,
        bounding_box=None,
        resolutions=None,
    ):
        self.data_format = data_format
        self.name = name
        self.category = category
        self.element_class = element_class
        self.num_channels = num_channels
        self.bounding_box = bounding_box or {
            "topLeft": (-1, -1, -1),
            "width": 0,
            "height": 0,
            "depth": 0,
        }
        self.wkw_resolutions = resolutions or []

    def to_json(self):
        return {
            "dataFormat": self.data_format,
            "name": self.name,
            "category": self.category,
            "elementClass": self.element_class,
            "num_channels": self.num_channels,
            "boundingBox": {}
            if self.bounding_box is None
            else {
                "topLeft": self.bounding_box["topLeft"],
                "width": self.bounding_box["width"],
                "height": self.bounding_box["height"],
                "depth": self.bounding_box["depth"],
            },
            "wkwResolutions": [r.to_json() for r in self.wkw_resolutions],
        }

    def get_element_type(self):
        return np.dtype(self.element_class)

    @classmethod
    def from_json(cls, json_data, resolution_type):
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["dataFormat"],
            json_data["name"],
            json_data["category"],
            json_data["elementClass"],
            json_data["num_channels"],
            json_data["boundingBox"],
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties.add_resolution(resolution_type.from_json(resolution))

        return layer_properties

    def add_resolution(self, resolution):
        self.wkw_resolutions.append(resolution)

    def delete_resolution(self, resolution):
        self.wkw_resolutions.delete(resolution)

    def get_bounding_box_size(self):
        return (
            self.bounding_box["width"],
            self.bounding_box["height"],
            self.bounding_box["depth"],
        )

    def get_bounding_box_offset(self):
        return tuple(self.bounding_box["topLeft"])

    def set_bounding_box_size(self, size):
        self.bounding_box["width"] = size[0]
        self.bounding_box["height"] = size[1]
        self.bounding_box["depth"] = size[2]

    def set_bounding_box_offset(self, offset):
        self.bounding_box["topLeft"] = offset

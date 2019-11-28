import json
from collections import namedtuple

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
        return {
            "resolution": self.mag.to_array()
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(json_data["resolution"])


class WkResolution(Resolution):

    def __init__(self, mag, cube_length):
        self.mag = Mag(mag)
        self.cube_length = cube_length

    def to_json(self):
        return {
            "resolution": self.mag.to_array(),
            "cube_length": self.cube_length
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(json_data["resolution"], json_data["cube_length"])


class Properties:

    # TODO: implement a test that reads a json and then writes it again and compares the files

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

    def add_layer(self, layer_name, category, element_class):
        pass

    def delete_layer(self, layer_name):
        del self.data_layers[layer_name]
        self.export_as_json()

    def add_mag(self, layer_name, mag, cube_length):
        pass

    def delete_mag(self, layer_name, mag):
        self.data_layers[layer_name].wkw_resolutions = [
            r for r in self.data_layers[layer_name].wkw_resolutions if r.mag != Mag(mag)
        ]
        self.export_as_json()


class WKProperties(Properties):

    @classmethod
    def from_json(cls, path):
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
            for layer in data['dataLayers']:
                data_layers[layer["name"]] = LayerProperties.from_json(layer, WkResolution)

            return cls(path, data['id']['name'], data['scale'], data['id']['team'], data_layers)

    def export_as_json(self):
        data = {
            'id': {
                'name': self.name,
                'team': self.team
            },
            'scale': self.scale,
            'dataLayers': [self.data_layers[layer_name].to_json() for layer_name in self.data_layers]
        }
        with open(self.path, 'w') as outfile:
            json.dump(data, outfile, indent=4, separators=(',', ': '))

    def add_layer(self, layer_name, category, element_class):
        new_layer = LayerProperties("wkw", layer_name, category, element_class)
        self.data_layers[layer_name] = new_layer
        self.export_as_json()

    def add_mag(self, layer_name, mag, cube_length):
        self.data_layers[layer_name].add_resolution(WkResolution(mag, cube_length))
        self.export_as_json()


class TiffProperties(Properties):

    def __init__(self, path, name, scale, team="", data_layers=None, grid_shape=(0, 0), min_dimensions=(0, 0), max_dimensions=(0, 0)):
        super().__init__(path, name, scale, team, data_layers)
        self.grid_shape = grid_shape
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions

    @classmethod
    def from_json(cls, path):
        with open(path) as datasource_properties:
            data = json.load(datasource_properties)

            # reconstruct data_layers
            data_layers = {}
            for layer in data['dataLayers']:
                data_layers[layer["name"]] = LayerProperties.from_json(layer, TiffResolution)

            return cls(
                path,
                data['id']['name'],
                data['scale'],
                data['id']['team'],
                data_layers,
                data['grid_shape'],
                data['min_dimensions'],
                data['max_dimensions']
            )

    def export_as_json(self):
        data = {
            'name': self.name,
            'team': self.team,
            'scale': self.scale,
            'dataLayers': [self.data_layers[layer_name].to_json() for layer_name in self.data_layers],
            'grid_shape': self.grid_shape,
            'min_dimensions': self.min_dimensions,
            'max_dimensions': self.max_dimensions
        }
        with open(self.path, 'w') as outfile:
            json.dump(data, outfile, indent=4, separators=(',', ': '))

    def add_layer(self, layer_name, category, element_class="uint8"):
        new_layer = LayerProperties("tiff", layer_name, category, element_class)
        self.data_layers[layer_name] = new_layer
        self.export_as_json()

    def add_mag(self, layer_name, mag, cube_length):
        self.data_layers[layer_name].add_resolution(TiffResolution(mag))
        self.export_as_json()


BoundingBox = namedtuple('BoundingBox', 'topLeft width height depth')


class LayerProperties:

    def __init__(self, data_format, name, category, element_class, bounding_box=None, resolutions=None):
        self.data_format = data_format
        self.name = name
        self.category = category
        self.element_class = element_class
        self.bounding_box = bounding_box
        self.wkw_resolutions = resolutions or []

    def to_json(self):
        return {
            "dataFormat": self.data_format,
            "name": self.name,
            "category": self.category,
            "elementClass": self.element_class,
            "boundingBox": {} if self.bounding_box is None else {
                "topLeft": self.bounding_box.topLeft,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
                "depth": self.bounding_box.depth
            },
            "wkwResolutions": [r.to_json() for r in self.wkw_resolutions]
        }

    @classmethod
    def from_json(cls, json_data, ResolutionType):
        # create LayerProperties without resolutions
        layer_properties = cls(
            json_data["dataFormat"],
            json_data["name"],
            json_data["category"],
            json_data["elementClass"],
            json_data["boundingBox"]
        )

        # add resolutions to LayerProperties
        for resolution in json_data["wkwResolutions"]:
            layer_properties.add_resolution(
                ResolutionType.from_json(resolution)
            )

        return layer_properties

    def add_resolution(self, resolution):
        self.wkw_resolutions.append(resolution)

    def delete_resolution(self, resolution):
        self.wkw_resolutions.delete(resolution)

    def set_boundingbox(self, bounding_box):  # TODO: maybe adjust this method to only "update" values
        self.bounding_box = bounding_box

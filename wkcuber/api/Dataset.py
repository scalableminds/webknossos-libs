from shutil import rmtree
from abc import ABC, abstractmethod
from os import mkdir
from os.path import join, normpath, basename
from pathlib import Path
import numpy as np

from wkcuber.api.Properties import WKProperties, TiffProperties
from wkcuber.api.Layer import Layer


class AbstractDataset(ABC):
    @abstractmethod
    def __init__(self, properties):
        self.layers = {}
        self.path = Path(properties.path).parent
        self.properties = properties

        # construct self.layer
        for layer_name in self.properties.data_layers:
            layer = self.properties.data_layers[layer_name]
            self.add_layer(
                layer.name, layer.category, layer.element_class, layer.num_channels
            )
            for resolution in layer.wkw_resolutions:
                try:
                    # fails if the resolution is of type TiffResolution, because Tiffs do not have a cube_size
                    self.layers[layer_name].setup_mag(
                        resolution.mag.to_layer_name(), resolution.cube_length
                    )
                except AttributeError:
                    self.layers[layer_name].setup_mag(resolution.mag.to_layer_name())

    @classmethod
    @abstractmethod
    def open(cls, path):
        pass

    @classmethod
    def create_with_properties(cls, properties):
        # initialize object
        dataset = cls(properties)
        # create directories on disk and write datasource-properties.json
        try:
            mkdir(dataset.path)
            dataset.properties.export_as_json()
        except OSError:
            raise FileExistsError("Creation of Dataset {} failed".format(dataset.path))

        return dataset

    @classmethod
    @abstractmethod
    def create(cls, path, scale):
        pass

    def downsample(self, layer, target_mag_shape, source_mag):
        raise NotImplemented()

    def get_properties(self):
        return self.properties

    def get_layer(self, layer_name) -> Layer:
        if layer_name not in self.layers.keys():
            raise IndexError(
                "The layer {} is not a layer of this dataset".format(layer_name)
            )
        return self.layers[layer_name]

    def add_layer(self, layer_name, category, dtype=np.dtype("uint8"), num_channels=1):
        # normalize the value of dtype in case the parameter was passed as a string
        dtype = np.dtype(dtype)

        if layer_name in self.layers.keys():
            raise IndexError(
                "Adding layer {} failed. There is already a layer with this name".format(
                    layer_name
                )
            )
        self.layers[layer_name] = Layer(layer_name, self, dtype, num_channels)
        self.properties.add_layer(layer_name, category, dtype.name, num_channels)
        return self.layers[layer_name]

    def delete_layer(self, layer_name):
        if layer_name not in self.layers.keys():
            raise IndexError(
                "Removing layer {} failed. There is no layer with this name".format(
                    layer_name
                )
            )
        del self.layers[layer_name]
        self.properties.delete_layer(layer_name)
        # delete files on disk
        rmtree(join(self.path, layer_name))


class WKDataset(AbstractDataset):
    @classmethod
    def open(cls, path):
        properties = WKProperties.from_json(join(path, "datasource-properties.json"))
        return cls(properties)

    @classmethod
    def create(cls, path, scale):
        name = basename(normpath(path))
        properties = WKProperties(join(path, "datasource-properties.json"), name, scale)
        return WKDataset.create_with_properties(properties)

    def __init__(self, properties):
        super().__init__(properties)
        assert isinstance(properties, WKProperties)

    def to_tiff_dataset(self, new_dataset_path):
        raise NotImplementedError  # TODO; implement


class TiffDataset(AbstractDataset):
    @classmethod
    def open(cls, path):
        properties = TiffProperties.from_json(join(path, "datasource-properties.json"))
        return cls(properties)

    @classmethod
    def create(cls, path, scale):
        name = basename(normpath(path))
        properties = TiffProperties(
            join(path, "datasource-properties.json"), name, scale
        )
        return TiffDataset.create_with_properties(properties)

    def __init__(self, properties):
        super().__init__(properties)
        assert isinstance(properties, TiffProperties)

    def to_wk_dataset(self, new_dataset_path):
        raise NotImplementedError  # TODO; implement

from shutil import rmtree
from abc import ABC, abstractmethod
from os import makedirs, path
from os.path import join, normpath, basename
from pathlib import Path
import numpy as np

from wkcuber.api.Properties import WKProperties, TiffProperties, Properties
from wkcuber.api.Layer import Layer, WKLayer, TiffLayer
from wkcuber.api.View import View


class AbstractDataset(ABC):
    @abstractmethod
    def __init__(self, dataset_path):
        properties = self._get_properties_type()._from_json(
            join(dataset_path, Properties.FILE_NAME)
        )
        self.layers = {}
        self.path = Path(properties.path).parent
        self.properties = properties

        # construct self.layer
        for layer_name in self.properties.data_layers:
            layer = self.properties.data_layers[layer_name]
            self.add_layer(
                layer.name, layer.category, layer.element_class, layer.num_channels
            )
            for resolution in layer.wkw_magnifications:
                self.layers[layer_name].setup_mag(resolution.mag.to_layer_name())

    @classmethod
    def create_with_properties(cls, properties):
        dataset_path = path.dirname(properties.path)
        # create directories on disk and write datasource-properties.json
        try:
            makedirs(dataset_path)
            properties._export_as_json()
        except OSError:
            raise FileExistsError("Creation of Dataset {} failed".format(dataset_path))

        # initialize object
        return cls(dataset_path)

    @classmethod
    @abstractmethod
    def create(cls, dataset_path, scale):
        pass

    def downsample(self, layer, target_mag_shape, source_mag):
        raise NotImplemented()

    def get_properties(self) -> Properties:
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
        self.layers[layer_name] = self._create_layer(layer_name, dtype, num_channels)
        self.properties._add_layer(layer_name, category, dtype.name, num_channels)
        return self.layers[layer_name]

    def get_or_add_layer(
        self, layer_name, category, dtype=None, num_channels=None
    ) -> Layer:
        if layer_name in self.layers.keys():
            assert self.properties.data_layers[layer_name].category == category, (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the categories do not match. "
                + f"The category of the existing layer is '{self.properties.data_layers[layer_name].category}' "
                + f"and the passed parameter is '{category}'."
            )
            assert dtype is None or self.layers[layer_name].dtype == np.dtype(dtype), (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the dtypes do not match. "
                + f"The dtype of the existing layer is '{self.layers[layer_name].dtype}' "
                + f"and the passed parameter is '{dtype}'."
            )
            assert (
                num_channels is None
                or self.layers[layer_name].num_channels == num_channels
            ), (
                f"Cannot get_or_add_layer: The layer '{layer_name}' already exists, but the number of channels do not match. "
                + f"The number of channels of the existing layer are '{self.layers[layer_name].num_channels}' "
                + f"and the passed parameter is '{num_channels}'."
            )
            return self.layers[layer_name]
        else:
            return self.add_layer(layer_name, category, dtype, num_channels)

    def delete_layer(self, layer_name):
        if layer_name not in self.layers.keys():
            raise IndexError(
                f"Removing layer {layer_name} failed. There is no layer with this name"
            )
        del self.layers[layer_name]
        self.properties._delete_layer(layer_name)
        # delete files on disk
        rmtree(join(self.path, layer_name))

    def get_view(self, layer_name, mag_name, size, global_offset=(0, 0, 0)) -> View:
        layer = self.get_layer(layer_name)
        mag = layer.get_mag(mag_name)
        mag_file_path = path.join(self.path, layer.name, mag.name)

        return mag.get_view(mag_file_path, size=size, global_offset=global_offset)

    def _create_layer(self, layer_name, dtype, num_channels) -> Layer:
        raise NotImplementedError

    @abstractmethod
    def _get_properties_type(self):
        pass


class WKDataset(AbstractDataset):
    @classmethod
    def create(cls, dataset_path, scale):
        name = basename(normpath(dataset_path))
        properties = WKProperties(join(dataset_path, Properties.FILE_NAME), name, scale)
        return WKDataset.create_with_properties(properties)

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        assert isinstance(self.properties, WKProperties)

    def to_tiff_dataset(self, new_dataset_path):
        raise NotImplementedError  # TODO; implement

    def _create_layer(self, layer_name, dtype, num_channels) -> Layer:
        return WKLayer(layer_name, self, dtype, num_channels)

    def _get_properties_type(self):
        return WKProperties


class TiffDataset(AbstractDataset):
    @classmethod
    def create(cls, dataset_path, scale):
        name = basename(normpath(dataset_path))
        properties = TiffProperties(
            join(dataset_path, Properties.FILE_NAME), name, scale
        )
        return TiffDataset.create_with_properties(properties)

    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        assert isinstance(self.properties, TiffProperties)

    def to_wk_dataset(self, new_dataset_path):
        raise NotImplementedError  # TODO; implement

    def _create_layer(self, layer_name, dtype, num_channels) -> Layer:
        return TiffLayer(layer_name, self, dtype, num_channels)

    def _get_properties_type(self):
        return TiffProperties

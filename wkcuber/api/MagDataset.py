from os.path import join

from wkw import wkw
import numpy as np

import wkcuber.api as api
from wkcuber.api.View import WKView, TiffView
from wkcuber.api.TiffData.TiffMag import TiffMagHeader


class MagDataset:
    def __init__(self, layer, name):
        self.layer = layer
        self.name = name
        self.header = self.get_header()

        file_path = join(self.layer.dataset.path, self.layer.name, self.name)
        size = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_size()
        offset = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_offset()
        self.view = self.get_view(
            file_path, size, global_offset=(0, 0, 0), is_bounded=False
        )

    def open(self):
        self.view.initialize()

    def close(self):
        self.view.close()

    def read(self, size, offset=(0, 0, 0)) -> np.array:
        return self.view.read(size, offset)

    def write(self, data, offset=(0, 0, 0)):
        self._assert_valid_num_channels(data.shape)
        self.view.write(data, offset)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset = layer_properties.get_bounding_box_offset()
        current_size = layer_properties.get_bounding_box_size()

        new_offset = (
            offset
            if current_offset == (-1, -1, -1)
            else tuple(min(x) for x in zip(current_offset, offset))
        )
        total_size = tuple(max(x) for x in zip(current_size, data.shape[-3:]))
        self.view.size = total_size

        self.layer.dataset.properties._set_bounding_box_of_layer(
            self.layer.name, new_offset, total_size
        )

    def get_header(self):
        raise NotImplementedError

    def get_view(self, mag_file_path, size, global_offset, is_bounded=True):
        raise NotImplementedError

    def _assert_valid_num_channels(self, write_data_shape):
        num_channels = self.layer.num_channels
        if len(write_data_shape) == 3:
            assert (
                num_channels == 1
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data (1)"
        else:
            assert (
                num_channels == 3
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data ({write_data_shape[0]})"


class WKMagDataset(MagDataset):
    def __init__(self, layer, name, block_len, file_len, block_type):
        self.block_len = block_len
        self.file_len = file_len
        self.block_type = block_type
        super().__init__(layer, name)

    def get_header(self) -> wkw.Header:
        return wkw.Header(
            voxel_type=self.layer.dtype,
            num_channels=self.layer.num_channels,
            version=1,
            block_len=self.block_len,
            file_len=self.file_len,
            block_type=self.block_type,
        )

    def get_view(self, mag_file_path, size, global_offset, is_bounded=True) -> WKView:
        return WKView(mag_file_path, self.header, size, global_offset, is_bounded)

    @classmethod
    def create(cls, layer, name, block_len, file_len, block_type):
        mag_dataset = cls(layer, name, block_len, file_len, block_type)
        wkw.Dataset.create(
            join(layer.dataset.path, layer.name, mag_dataset.name), mag_dataset.header
        )

        return mag_dataset


class TiffMagDataset(MagDataset):
    def get_header(self) -> TiffMagHeader:
        return TiffMagHeader(
            dtype=self.layer.dtype, num_channels=self.layer.num_channels
        )

    def get_view(self, mag_file_path, size, global_offset, is_bounded=True) -> TiffView:
        return TiffView(mag_file_path, self.header, size, global_offset, is_bounded)

    @classmethod
    def create(cls, layer, name):
        mag_dataset = cls(layer, name)
        return mag_dataset

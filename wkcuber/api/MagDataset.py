from os.path import join

from wkw import wkw
import numpy as np

import wkcuber.api as api
from wkcuber.api.View import WKView, TiffView
from wkcuber.api.TiffData.TiffMag import TiffMagHeader
from wkcuber.mag import Mag


class MagDataset:
    def __init__(self, layer, name):
        self.layer = layer
        self.name = name
        self.header = self.get_header()

        self.view = self.get_view(is_bounded=False)

    def open(self):
        self.view.open()

    def close(self):
        self.view.close()

    def read(self, size, offset=(0, 0, 0)) -> np.array:
        return self.view.read(size, offset)

    def write(self, data, offset=(0, 0, 0)):
        self._assert_valid_num_channels(data.shape)
        self.view.write(data, offset)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset_in_mag1 = layer_properties.get_bounding_box_offset()
        current_size_in_mag1 = layer_properties.get_bounding_box_size()

        mag = Mag(self.name)
        mag_np = mag.as_np()

        offset_in_mag1 = tuple(np.array(offset) * mag_np)

        new_offset_in_mag1 = (
            offset_in_mag1
            if current_offset_in_mag1 == (-1, -1, -1)
            else tuple(min(x) for x in zip(current_offset_in_mag1, offset_in_mag1))
        )

        old_end_offset_in_mag1 = np.array(current_offset_in_mag1) + np.array(
            current_size_in_mag1
        )
        new_end_offset_in_mag1 = (np.array(offset) + np.array(data.shape[-3:])) * mag_np
        max_end_offset_in_mag1 = np.array(
            [old_end_offset_in_mag1, new_end_offset_in_mag1]
        ).max(axis=0)
        total_size_in_mag1 = max_end_offset_in_mag1 - np.array(new_offset_in_mag1)
        total_size = total_size_in_mag1 / mag_np

        self.view.size = tuple(total_size)

        self.layer.dataset.properties._set_bounding_box_of_layer(
            self.layer.name, tuple(new_offset_in_mag1), tuple(total_size_in_mag1)
        )

    def get_header(self):
        raise NotImplementedError

    def get_view(self, size=None, offset=None, is_bounded=True):
        size_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_size()

        if offset is None:
            offset = (0, 0, 0)

        if size is None:
            size = size_in_properties

        # assert that the parameter size is valid
        for s1, s2, off in zip(size_in_properties, size, offset):
            if s2 + off > s1 and is_bounded:
                raise AssertionError(
                    f"The combination of the passed parameter 'size' {size} and {offset} are not compatible with the "
                    f"size ({size_in_properties}) from the properties.json."
                )

        mag_file_path = join(self.layer.dataset.path, self.layer.name, self.name)
        offset_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_offset()
        dataset_offset = (
            (0, 0, 0) if offset_in_properties == (-1, -1, -1) else offset_in_properties
        )
        global_offset = np.array(dataset_offset) + np.array(offset)
        return self._get_view_type()(
            mag_file_path, self.header, size, global_offset, is_bounded
        )

    def _get_view_type(self):
        raise NotImplementedError

    def _assert_valid_num_channels(self, write_data_shape):
        num_channels = self.layer.num_channels
        if len(write_data_shape) == 3:
            assert (
                num_channels == 1
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data (1)"
        else:
            assert (
                num_channels == write_data_shape[0]
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

    @classmethod
    def create(cls, layer, name, block_len, file_len, block_type):
        mag_dataset = cls(layer, name, block_len, file_len, block_type)
        wkw.Dataset.create(
            join(layer.dataset.path, layer.name, mag_dataset.name), mag_dataset.header
        )

        return mag_dataset

    def _get_view_type(self):
        return WKView


class TiffMagDataset(MagDataset):
    def __init__(self, layer, name, pattern):
        self.pattern = pattern
        super().__init__(layer, name)

    def get_header(self) -> TiffMagHeader:
        return TiffMagHeader(
            pattern=self.pattern,
            dtype=self.layer.dtype,
            num_channels=self.layer.num_channels,
            tile_size=self.layer.dataset.properties.tile_size,
        )

    @classmethod
    def create(cls, layer, name, pattern):
        mag_dataset = cls(layer, name, pattern)
        return mag_dataset

    def _get_view_type(self):
        return TiffView


class TiledTiffMagDataset(TiffMagDataset):
    def get_tile(self, x_index, y_index, z_index) -> np.array:
        tile_size = self.layer.dataset.properties.tile_size
        size = tuple(tile_size) + tuple((1,))
        offset = np.array((0, 0, 0)) + np.array(size) * np.array(
            (x_index, y_index, z_index)
        )
        return self.read(size, offset)

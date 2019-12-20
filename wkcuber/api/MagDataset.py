from os.path import join

from wkw import wkw

import wkcuber.api as api
from wkcuber.api.Slice import WKSlice, TiffSlice
from wkcuber.api.TiffData.TiffMag import TiffMagHeader


class MagDataset:
    def __init__(self, layer, name):
        self.mag = {}
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
        self.slice = self.__get_slice_type__()(
            file_path, self.header, size, global_offset=(0, 0, 0)
        )

        # MagDataset uses Slice to access the actual data, but there are no bounds for an MagDataset (unlike a Slice)
        self.slice.is_bounded = False

    def open(self):
        self.slice.open()

    def close(self):
        self.slice.close()

    def read(self, size, offset=(0, 0, 0)):
        return self.slice.read(size, offset)

    def write(self, data, offset=(0, 0, 0)):
        self.__assert_valid_num_channels__(data.shape)
        self.slice.write(data, offset)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset = layer_properties.get_bounding_box_offset()
        current_size = layer_properties.get_bounding_box_size()

        if self.slice.check_bounds(current_offset, current_size):
            new_offset = (
                offset
                if current_offset == (-1, -1, -1)
                else tuple(min(x) for x in zip(current_offset, offset))
            )
            total_size = tuple(max(x) for x in zip(current_size, data.shape[-3:]))
            self.slice.size = total_size
            # no need to update the offset of the slice, because it is always the absolute offset (0, 0, 0)

            self.layer.dataset.properties.set_bounding_box_size_of_layer(
                self.layer.name, total_size
            )
            self.layer.dataset.properties.set_bounding_box_offset_of_layer(
                self.layer.name, new_offset
            )

    def get_header(self):
        raise NotImplementedError

    def get_slice(self, mag_file_path, size, global_offset):
        raise NotImplementedError

    def __get_slice_type__(self):
        raise NotImplementedError

    def __assert_valid_num_channels__(self, write_data_shape):
        num_channels = self.layer.num_channels
        if len(write_data_shape) == 3:
            assert num_channels == 1, (
                "The number of channels of the dataset (%d) does not match the number of channels of the passed data (%d)"
                % (num_channels, 1)
            )
        else:
            assert num_channels == 3, (
                "The number of channels of the dataset (%d) does not match the number of channels of the passed data (%d)"
                % (num_channels, write_data_shape[0])
            )


class WKMagDataset(MagDataset):
    def __init__(self, layer, name, block_len, file_len, block_type):
        self.block_len = block_len
        self.file_len = file_len
        self.block_type = block_type
        super().__init__(layer, name)

    def get_header(self):
        return wkw.Header(
            voxel_type=self.layer.dtype,
            num_channels=self.layer.num_channels,
            version=1,
            block_len=self.block_len,
            file_len=self.file_len,
            block_type=self.block_type,
        )

    def get_slice(self, mag_file_path, size, global_offset):
        return WKSlice(mag_file_path, self.get_header(), size, global_offset)

    def __get_slice_type__(self):
        return WKSlice

    @classmethod
    def create(cls, layer, name, block_len, file_len, block_type):
        mag_dataset = cls(layer, name, block_len, file_len, block_type)
        wkw.Dataset.create(
            join(layer.dataset.path, layer.name, mag_dataset.name), mag_dataset.header
        )

        return mag_dataset


class TiffMagDataset(MagDataset):
    def get_header(self):
        return TiffMagHeader(
            dtype=self.layer.dtype, num_channels=self.layer.num_channels
        )

    def get_slice(self, mag_file_path, size, global_offset):
        return TiffSlice(mag_file_path, self.get_header(), size, global_offset)

    def __get_slice_type__(self):
        return TiffSlice

    @classmethod
    def create(cls, layer, name):
        mag_dataset = cls(layer, name)
        return mag_dataset

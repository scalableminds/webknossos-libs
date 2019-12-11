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

        file_path = join(self.layer.dataset.path, self.layer.name, self.name)
        size = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_size()
        offset = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_offset()
        if isinstance(self.layer.dataset, api.Dataset.WKDataset):
            self.header = wkw.Header(
                voxel_type=self.layer.dtype, num_channels=self.layer.num_channels
            )
            self.slice = WKSlice(file_path, size, offset)
        elif isinstance(self.layer.dataset, api.Dataset.TiffDataset):
            self.slice = TiffSlice(file_path, size, offset)
            self.header = TiffMagHeader(
                dtype=self.layer.dtype, num_channels=self.layer.num_channels
            )
        else:
            raise TypeError()

        # MagDataset uses Slice to access the actual data, but there are no bounds for an MagDataset (unlike a Slice)
        self.slice.is_bounded = False

    def open(self):
        self.slice.open(self.header)

    def close(self):
        self.slice.close()

    def read(self, size, offset=(0, 0, 0)):
        return self.slice.read(size, offset, self.header)

    def write(self, data, offset=(0, 0, 0)):
        self.slice.write(data, offset, self.header)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset = layer_properties.get_bounding_box_offset()
        current_size = layer_properties.get_bounding_box_size()

        if self.slice.check_bounds(current_offset, current_size):
            # total_offset = tuple(sum(x) for x in zip(, offset)) # TODO: can the offset change ?
            total_size = tuple(max(x) for x in zip(current_size, data.shape[-3:]))
            self.slice.size = total_size
            self.layer.dataset.properties.set_bounding_box_size_of_layer(
                self.layer.name, total_size
            )

    @classmethod
    def create(cls, layer, name):
        mag_dataset = cls(layer, name)
        if isinstance(layer.dataset, api.Dataset.WKDataset):
            wkw.Dataset.create(
                join(layer.dataset.path, layer.name, mag_dataset.name),
                mag_dataset.header,
            )

        return mag_dataset

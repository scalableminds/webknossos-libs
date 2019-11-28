from os.path import join

import wkcuber.api as api
from wkcuber.api.Slice import WKSlice, TiffSlice


class MagDataset:

    def __init__(self, layer, name):
        self.mag = {}
        self.layer = layer
        self.name = name

        file_path = join(self.layer.dataset.path, self.layer.name, self.name)
        if isinstance(self.layer.dataset, api.Dataset.WKDataset):
            self.slice = WKSlice(file_path)
        elif isinstance(self.layer.dataset, api.Dataset.TiffDataset):
            self.slice = TiffSlice(file_path)
        else:
            raise TypeError()

        # MagDataset uses Slice to access the actual data, but there are no bounds for an MagDataset (unlike a Slice)
        self.slice.is_bounded = False

    def open(self):
        self.slice.open()

    def close(self):
        self.slice.close()

    def read(self, size, offset=(0, 0, 0)):
        return self.slice.read(size, offset)

    def write(self, data, offset=None):
        self.slice.write(data, offset)
        # TODO: potentially update slice.bounding_box
        # TODO: potentially update properties.json

import numpy as np
from os import path
from PIL import Image

from .vendor.dm3 import DM3


class PillowImageReader:
    def read_array(self, file_name, dtype):
        this_layer = np.array(Image.open(file_name), np.dtype(dtype))
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name):
        with Image.open(file_name) as test_img:
            return (test_img.width, test_img.height)


class Dm3ImageReader:
    def read_array(self, file_name, dtype):
        this_layer = DM3(file_name).imagedata.astype(np.dtype(dtype))
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name):
        test_img = DM3(file_name)
        return (test_img.width, test_img.height)


class ImageReader:
    def __init__(self):
        self.readers = {
            ".tif": PillowImageReader(),
            ".tiff": PillowImageReader(),
            ".jpg": PillowImageReader(),
            ".jpeg": PillowImageReader(),
            ".png": PillowImageReader(),
            ".dm3": Dm3ImageReader(),
        }

    def read_array(self, file_name, dtype):
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_array(file_name, dtype)

    def read_dimensions(self, file_name):
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dimensions(file_name)


image_reader = ImageReader()

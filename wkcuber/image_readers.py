import numpy as np
import logging
from os import path
from PIL import Image

from .vendor.dm3 import DM3
from .vendor.dm4 import DM4File

# Disable PIL's maximum image limit.
Image.MAX_IMAGE_PIXELS = None


class PillowImageReader:
    def read_array(self, file_name, dtype):
        this_layer = np.array(Image.open(file_name), np.dtype(dtype))
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name):
        with Image.open(file_name) as test_img:
            return (test_img.width, test_img.height)

    def read_channel_count(self, file_name):
        with Image.open(file_name) as test_img:
            this_layer = np.array(test_img)
            if this_layer.ndim == 2:
                # For two-dimensional data, the channel count is one
                return 1
            else:
                return this_layer.shape[-1]


def to_target_datatype(data: np.ndarray, target_dtype) -> np.ndarray:

    factor = (1 + np.iinfo(data.dtype).max) / (1 + np.iinfo(target_dtype).max)
    return (data / factor).astype(np.dtype(target_dtype))


class Dm3ImageReader:
    def read_array(self, file_name, dtype):
        dm3_file = DM3(file_name)
        this_layer = to_target_datatype(dm3_file.imagedata, dtype)
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name):
        test_img = DM3(file_name)
        return (test_img.width, test_img.height)

    def read_channel_count(self, file_name):
        logging.info("Assuming single channel for DM3 data")
        return 1


class Dm4ImageReader:
    def _read_tags(self, dm4file):

        tags = dm4file.read_directory()
        image_data_tag = (
            tags.named_subdirs["ImageList"]
            .unnamed_subdirs[1]
            .named_subdirs["ImageData"]
        )
        image_tag = image_data_tag.named_tags["Data"]

        return image_data_tag, image_tag

    def _read_dimensions(self, dm4file, image_data_tag):

        width = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[0]
        )
        height = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[1]
        )
        return width, height

    def read_array(self, file_name, dtype):

        dm4file = DM4File.open(file_name)
        image_data_tag, image_tag = self._read_tags(dm4file)
        width, height = self._read_dimensions(dm4file, image_data_tag)

        data = np.array(dm4file.read_tag_data(image_tag), dtype=np.uint16)

        data = data.reshape((width, height)).T
        data = np.expand_dims(data, 2)
        data = to_target_datatype(data, dtype)

        dm4file.close()

        return data

    def read_dimensions(self, file_name):

        dm4file = DM4File.open(file_name)
        image_data_tag, _ = self._read_tags(dm4file)
        dimensions = self._read_dimensions(dm4file, image_data_tag)
        dm4file.close()

        return dimensions

    def read_channel_count(self, file_name):
        logging.info("Assuming single channel for DM4 data")
        return 1


class ImageReader:
    def __init__(self):
        self.readers = {
            ".tif": PillowImageReader(),
            ".tiff": PillowImageReader(),
            ".jpg": PillowImageReader(),
            ".jpeg": PillowImageReader(),
            ".png": PillowImageReader(),
            ".dm3": Dm3ImageReader(),
            ".dm4": Dm4ImageReader(),
        }

    def read_array(self, file_name, dtype):
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_array(file_name, dtype)

    def read_dimensions(self, file_name):
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dimensions(file_name)

    def read_channel_count(self, file_name):
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_channel_count(file_name)


image_reader = ImageReader()

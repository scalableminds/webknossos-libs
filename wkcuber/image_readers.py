from typing import Tuple, Dict, Union

import numpy as np
import logging
from os import path
from PIL import Image

from .vendor.dm3 import DM3
from .vendor.dm4 import DM4File, DM4TagHeader
from tifffile import TiffFile

# Disable PIL's maximum image limit.
Image.MAX_IMAGE_PIXELS = None


class ImageReader:
    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        pass

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        pass

    def read_channel_count(self, file_name: str) -> int:
        pass

    def read_z_slices_per_file(
        self, file_name: str  # pylint: disable=unused-argument
    ) -> int:
        return 1


class PillowImageReader(ImageReader):
    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        this_layer = np.array(Image.open(file_name), dtype)
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        with Image.open(file_name) as test_img:
            return test_img.width, test_img.height

    def read_channel_count(self, file_name: str) -> int:
        with Image.open(file_name) as test_img:
            this_layer = np.array(test_img)
            if this_layer.ndim == 2:
                # For two-dimensional data, the channel count is one
                return 1
            else:
                return this_layer.shape[-1]  # pylint: disable=unsubscriptable-object


def to_target_datatype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    factor = (1 + np.iinfo(data.dtype).max) / (1 + np.iinfo(target_dtype).max)
    return (data / factor).astype(target_dtype)


class Dm3ImageReader(ImageReader):
    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        dm3_file = DM3(file_name)
        this_layer = to_target_datatype(dm3_file.imagedata, dtype)
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        test_img = DM3(file_name)
        return test_img.width, test_img.height

    def read_channel_count(self, _file_name: str) -> int:
        logging.info("Assuming single channel for DM3 data")
        return 1


class Dm4ImageReader(ImageReader):
    def _read_tags(self, dm4file: DM4File) -> Tuple[DM4File.DM4TagDir, DM4TagHeader]:
        tags = dm4file.read_directory()
        image_data_tag = (
            tags.named_subdirs["ImageList"]
            .unnamed_subdirs[1]
            .named_subdirs["ImageData"]
        )
        image_tag = image_data_tag.named_tags["Data"]

        return image_data_tag, image_tag

    def _read_dimensions(
        self, dm4file: DM4File, image_data_tag: DM4File.DM4TagDir
    ) -> Tuple[int, int]:
        width = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[0]
        )
        height = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[1]
        )
        return width, height

    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        dm4file = DM4File.open(file_name)
        image_data_tag, image_tag = self._read_tags(dm4file)
        width, height = self._read_dimensions(dm4file, image_data_tag)

        data = np.array(dm4file.read_tag_data(image_tag), dtype=np.uint16)

        data = data.reshape((width, height)).T
        data = np.expand_dims(data, 2)
        data = to_target_datatype(data, dtype)

        dm4file.close()

        return data

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        dm4file = DM4File.open(file_name)
        image_data_tag, _ = self._read_tags(dm4file)
        dimensions = self._read_dimensions(dm4file, image_data_tag)
        dm4file.close()

        return dimensions

    def read_channel_count(self, _file_name: str) -> int:
        logging.info("Assuming single channel for DM4 data")
        return 1


def find_count_of_axis(tif_file: TiffFile, axis: str) -> int:
    assert len(tif_file.series) == 1, "only single tif series are supported"
    tif_series = tif_file.series[0]
    index = tif_series.axes.find(axis)
    if index == -1:
        return 1
    else:
        return tif_series.shape[index]  # pylint: disable=unsubscriptable-object


class TiffImageReader(ImageReader):
    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        with TiffFile(file_name) as tif_file:
            num_channels = self.read_channel_count(file_name)
            if len(tif_file.pages) > num_channels:
                data = np.array(
                    list(
                        map(
                            lambda x: x.asarray(),
                            tif_file.pages[
                                z_slice * num_channels : z_slice * num_channels
                                + num_channels
                            ],
                        )
                    ),
                    dtype,
                )
            else:
                data = np.array(
                    list(map(lambda x: x.asarray(), tif_file.pages[0:num_channels])),
                    dtype,
                )
            # transpose data to shape(x, y, channel_count)
            data = np.transpose(
                data,
                (
                    tif_file.pages[0].axes.find("X") + 1,
                    tif_file.pages[0].axes.find("Y") + 1,
                    0,
                ),
            )
            data = data.reshape(data.shape + (1,))
            return data

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        with TiffFile(file_name) as tif_file:
            return find_count_of_axis(tif_file, "X"), find_count_of_axis(tif_file, "Y")

    def read_channel_count(self, file_name: str) -> int:
        with TiffFile(file_name) as tif_file:
            return find_count_of_axis(tif_file, "C")

    def read_z_slices_per_file(self, file_name: str) -> int:
        with TiffFile(file_name) as tif_file:
            return find_count_of_axis(tif_file, "Z")


class ImageReaderManager:
    def __init__(self) -> None:
        self.readers: Dict[
            str,
            Union[TiffImageReader, PillowImageReader, Dm3ImageReader, Dm4ImageReader],
        ] = {
            ".tif": TiffImageReader(),
            ".tiff": TiffImageReader(),
            ".jpg": PillowImageReader(),
            ".jpeg": PillowImageReader(),
            ".png": PillowImageReader(),
            ".dm3": Dm3ImageReader(),
            ".dm4": Dm4ImageReader(),
        }

    def read_array(self, file_name: str, dtype: np.dtype, z_slice: int) -> np.ndarray:
        _, ext = path.splitext(file_name)

        # Image shape will be (x, y, channel_count, z=1) or (x, y, z=1)
        image = self.readers[ext].read_array(file_name, dtype, z_slice)
        # Standardize the image shape to (x, y, channel_count, z=1)
        if image.ndim == 3:
            image = image.reshape(image.shape + (1,))

        return image

    def read_dimensions(self, file_name: str) -> Tuple[int, int]:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dimensions(file_name)

    def read_channel_count(self, file_name: str) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_channel_count(file_name)

    def read_z_slices_per_file(self, file_name: str) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_z_slices_per_file(file_name)


image_reader = ImageReaderManager()

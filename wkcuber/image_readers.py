from pathlib import Path
from typing import Tuple, Dict, Union, Optional

import numpy as np
import logging
from os import path
from PIL import Image

from .vendor.dm3 import DM3
from .vendor.dm4 import DM4File, DM4TagHeader
from tifffile import TiffFile
from czifile import CziFile

# Disable PIL's maximum image limit.
Image.MAX_IMAGE_PIXELS = None


class ImageReader:
    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        pass

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        pass

    def read_channel_count(self, file_name: Path) -> int:
        pass

    def read_z_slices_per_file(
        self, file_name: Path  # pylint: disable=unused-argument
    ) -> int:
        return 1

    def read_dtype(self, file_name: Path) -> str:
        raise NotImplementedError()


class PillowImageReader(ImageReader):
    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        this_layer = np.array(Image.open(file_name), dtype)
        this_layer = this_layer.swapaxes(0, 1)
        if channel_index is not None:
            this_layer = this_layer[:, :, channel_index]
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        with Image.open(file_name) as test_img:
            return test_img.width, test_img.height

    def read_channel_count(self, file_name: Path) -> int:
        with Image.open(file_name) as test_img:
            this_layer = np.array(test_img)
            if this_layer.ndim == 2:
                # For two-dimensional data, the channel count is one
                return 1
            else:
                return this_layer.shape[-1]  # pylint: disable=unsubscriptable-object

    def read_dtype(self, file_name: Path) -> str:
        return np.array(Image.open(file_name)).dtype.name


def to_target_datatype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    factor = (1 + np.iinfo(data.dtype).max) / (1 + np.iinfo(target_dtype).max)
    return (data / factor).astype(target_dtype)


class Dm3ImageReader(ImageReader):
    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        dm3_file = DM3(file_name)
        this_layer = to_target_datatype(dm3_file.imagedata, dtype)
        this_layer = this_layer.swapaxes(0, 1)
        this_layer = this_layer.reshape(this_layer.shape + (1,))
        return this_layer

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        test_img = DM3(file_name)
        return test_img.width, test_img.height

    def read_channel_count(self, _file_name: Path) -> int:
        logging.info("Assuming single channel for DM3 data")
        return 1

    def read_dtype(self, file_name: Path) -> str:
        return DM3(file_name).imagedata.dtype.name


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

    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        dm4file = DM4File.open(str(file_name))
        image_data_tag, image_tag = self._read_tags(dm4file)
        width, height = self._read_dimensions(dm4file, image_data_tag)

        data = np.array(dm4file.read_tag_data(image_tag), dtype=np.uint16)

        data = data.reshape((width, height)).T
        data = np.expand_dims(data, 2)
        data = to_target_datatype(data, dtype)

        dm4file.close()

        return data

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        dm4file = DM4File.open(file_name)
        image_data_tag, _ = self._read_tags(dm4file)
        dimensions = self._read_dimensions(dm4file, image_data_tag)
        dm4file.close()

        return dimensions

    def read_channel_count(self, _file_name: Path) -> int:
        logging.info("Assuming single channel for DM4 data")
        return 1

    def read_dtype(self, file_name: Path) -> str:  # pylint: disable=unused-argument
        # DM4 standard input type is uint16
        return "uint16"


class TiffImageReader(ImageReader):
    def __init__(self) -> None:
        self.is_page_multi_channel: Optional[bool] = None
        self.num_pages_for_all_channels: Optional[int] = None

    @staticmethod
    def find_count_of_axis(tif_file: TiffFile, axis: str) -> int:
        assert len(tif_file.series) == 1, "only single tif series are supported"
        tif_series = tif_file.series[0]
        index = tif_series.axes.find(axis)
        if index == -1:
            return 1
        else:
            return tif_series.shape[index]  # pylint: disable=unsubscriptable-object

    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        with TiffFile(file_name) as tif_file:
            # We are caching 'num_pages_for_all_channels' and 'is_page_multi_channel'
            # because they are the same for all tiffs
            if self.num_pages_for_all_channels is None:
                self.num_pages_for_all_channels = self.read_channel_count(file_name)
            if self.is_page_multi_channel is None:
                self.is_page_multi_channel = tif_file.pages[0].ndim == 3
                if self.is_page_multi_channel:
                    self.num_pages_for_all_channels = 1

            channel_selected = channel_index is not None
            num_pages_in_result = (
                1 if channel_selected else self.num_pages_for_all_channels
            )

            # we need to set the channel_offset for multi-channel pages
            # because reading will fail otherwise and we handle the channel selection elsewhere
            channel_offset = (
                0
                if channel_index is None or self.is_page_multi_channel
                else channel_index
            )

            if len(tif_file.pages) > self.num_pages_for_all_channels:
                page_index = z_slice * self.num_pages_for_all_channels + channel_offset
                # single multi-page input file
                data = np.array(
                    list(
                        map(
                            lambda x: x.asarray(),
                            tif_file.pages[
                                page_index : page_index + num_pages_in_result
                            ],
                        )
                    ),
                    dtype,
                )
            else:
                # multiple input files
                data = np.array(
                    list(
                        map(
                            lambda x: x.asarray(),
                            tif_file.pages[
                                channel_offset : channel_offset + num_pages_in_result
                            ],
                        )
                    ),
                    dtype,
                )

            # if the pages are multi-channel, then we'll have 4 dimensions here because of [x:x+1] notation, so we select the data here
            if self.is_page_multi_channel:
                data = data[0]
                x_index = tif_file.pages[0].axes.find("X")
                y_index = tif_file.pages[0].axes.find("Y")
                c_index = tif_file.pages[0].axes.find("S")
            else:
                # if each page is a channel, there is no c_index in the page axes and through our selection the c_index is always 0 and therefore the other indices have to be incremented
                c_index = 0
                x_index = tif_file.pages[0].axes.find("X") + 1
                y_index = tif_file.pages[0].axes.find("Y") + 1

            # transpose data to shape(x, y, channel_count)
            data = data.transpose((x_index, y_index, c_index))
            # if page is multi-channel and one channel is selected, slice channel here. Resulting Format will be (X, Y), but the following reshape will fix this
            if self.is_page_multi_channel and channel_selected:
                data = data[:, :, channel_index]
            data = data.reshape(data.shape + (1,))
            return data

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        with TiffFile(file_name) as tif_file:
            return (
                TiffImageReader.find_count_of_axis(tif_file, "X"),
                TiffImageReader.find_count_of_axis(tif_file, "Y"),
            )

    def read_channel_count(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            c_count = TiffImageReader.find_count_of_axis(tif_file, "C")
            s_count = TiffImageReader.find_count_of_axis(tif_file, "S")
            assert not (
                c_count > 1 and s_count > 1
            ), "This file format is currently not supported."
            if s_count > 1:
                return s_count
            else:
                return c_count

    def read_z_slices_per_file(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            i_count = TiffImageReader.find_count_of_axis(tif_file, "I")
            if i_count > 1:
                return i_count
            return TiffImageReader.find_count_of_axis(tif_file, "Z")

    def read_dtype(self, file_name: Path) -> str:
        with TiffFile(file_name) as tif_file:
            return tif_file.series[  # pylint: disable=unsubscriptable-object
                0
            ].dtype.name


class CziImageReader(ImageReader):
    def __init__(self) -> None:
        self.tile_shape = None

    @staticmethod
    def find_count_of_axis(czi_file: CziFile, axis: str) -> int:
        index = czi_file.axes.find(axis)
        if index == -1:
            return 1
        else:
            return czi_file.shape[index]

    # returns format (X, Y)
    def _read_array_single_channel(
        self, czi_file: CziFile, channel_slice: int, dtype: np.dtype, z_slice: int
    ) -> np.ndarray:
        channel_index = czi_file.axes.find("C")
        x_index = czi_file.axes.find("X")
        y_index = czi_file.axes.find("Y")
        z_index = czi_file.axes.find("Z")
        channel_file_start = czi_file.start[channel_index]
        z_file_start = czi_file.start[z_index]
        for entry in czi_file.filtered_subblock_directory:
            if (
                entry.start[z_index] - z_file_start == z_slice
                and entry.start[channel_index] - channel_file_start == channel_slice
            ):
                # This case assumes that the data-segment contains a single channel and a single z slice
                data = entry.data_segment().data()
                # We are not sure if the order of the X and Y dimensions is always the same, so we check that we always produce the correct output format
                if x_index > y_index:
                    data = data.reshape(data.shape[y_index], data.shape[x_index])
                    data = data.swapaxes(0, 1)
                else:
                    data = data.reshape(data.shape[x_index], data.shape[y_index])
                data = to_target_datatype(data, dtype)
                return data

    # return format will be (X, Y, channel_count)
    def _read_array_all_channels(
        self, czi_file: CziFile, dtype: np.dtype, z_slice: int
    ) -> np.ndarray:
        # There can be a lot of axes in the czi_file, but we are only interested in the x, y and c indices
        x_index = czi_file.axes.find("X")
        y_index = czi_file.axes.find("Y")
        z_index = czi_file.axes.find("Z")
        c_index = czi_file.axes.find("C")
        indices = [(x_index, "X"), (y_index, "Y"), (c_index, "C")]
        # We are not sure, which ordering these indices have, so we order them to correctly reshape the data
        indices.sort()

        z_start = czi_file.start[z_index]
        for entry in czi_file.filtered_subblock_directory:
            if entry.start[z_index] - z_start == z_slice:
                data = entry.data_segment().data()
                # Reshaping the data to the shape of the selected axes from above
                data = data.reshape(
                    data.shape[indices[0][0]],
                    data.shape[indices[1][0]],
                    data.shape[indices[2][0]],
                )
                # After reshaping the data to the ordered indices, we now change the format to (X, Y, channel_count)
                data = np.transpose(
                    data,
                    (
                        [i for i, d in enumerate(indices) if d[1] == "X"][0],
                        [i for i, d in enumerate(indices) if d[1] == "Y"][0],
                        [i for i, d in enumerate(indices) if d[1] == "C"][0],
                    ),
                )
                data = to_target_datatype(data, dtype)
                return data

    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int],
    ) -> np.ndarray:
        with CziFile(file_name) as czi_file:
            c_index = czi_file.axes.find("C")  # pylint: disable=no-member
            channel_count = self.read_channel_count(file_name)
            # we assume the tile shape is constant, so we can cache it
            if self.tile_shape is None:
                self.tile_shape = (
                    czi_file.filtered_subblock_directory[  # pylint: disable=unsubscriptable-object
                        0
                    ]
                    .data_segment()
                    .data()
                    .shape
                )

            assert self.tile_shape is not None, "Cannot read tile shape format."

            # we assume either all channel are in one tile or each tile is single channel
            if self.tile_shape[c_index] != channel_count:
                if channel_index is not None:
                    data = self._read_array_single_channel(
                        czi_file, channel_index, dtype, z_slice
                    )
                    data = data.reshape(data.shape + (1,))
                    return data
                else:
                    x_count, y_count = self.read_dimensions(file_name)
                    output = np.empty((channel_count, x_count, y_count), dtype)
                    # format is (channel_count, x, y)
                    for i in range(0, channel_count):
                        output[i] = self._read_array_single_channel(
                            czi_file, i, dtype, z_slice
                        )

                    # transpose format to x, y, channel_count
                    output = np.transpose(output, (1, 2, 0))
                    return output
            else:
                data = self._read_array_all_channels(czi_file, dtype, z_slice)
                if channel_index is not None:
                    data = data[:, :, channel_index : channel_index + 1]
                return data

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        with CziFile(file_name) as czi_file:
            return (
                CziImageReader.find_count_of_axis(czi_file, "X"),
                CziImageReader.find_count_of_axis(czi_file, "Y"),
            )

    def read_channel_count(self, file_name: Path) -> int:
        with CziFile(file_name) as czi_file:
            return CziImageReader.find_count_of_axis(czi_file, "C")

    def read_z_slices_per_file(self, file_name: Path) -> int:
        with CziFile(file_name) as czi_file:
            return CziImageReader.find_count_of_axis(czi_file, "Z")

    def read_dtype(self, file_name: Path) -> str:
        with CziFile(file_name) as czi_file:
            return czi_file.dtype.name  # pylint: disable=no-member


class ImageReaderManager:
    def __init__(self) -> None:
        self.readers: Dict[
            str,
            Union[
                TiffImageReader,
                PillowImageReader,
                Dm3ImageReader,
                Dm4ImageReader,
                CziImageReader,
            ],
        ] = {
            ".tif": TiffImageReader(),
            ".tiff": TiffImageReader(),
            ".jpg": PillowImageReader(),
            ".jpeg": PillowImageReader(),
            ".png": PillowImageReader(),
            ".dm3": Dm3ImageReader(),
            ".dm4": Dm4ImageReader(),
            ".czi": CziImageReader(),
        }

    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        channel_index: Optional[int] = None,
    ) -> np.ndarray:
        _, ext = path.splitext(file_name)

        # Image shape will be (x, y, channel_count, z=1) or (x, y, z=1)
        image = self.readers[ext].read_array(file_name, dtype, z_slice, channel_index)
        # Standardize the image shape to (x, y, channel_count, z=1)
        if image.ndim == 3:
            image = image.reshape(image.shape + (1,))

        return image

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dimensions(file_name)

    def read_channel_count(self, file_name: Path) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_channel_count(file_name)

    def read_z_slices_per_file(self, file_name: Path) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_z_slices_per_file(file_name)

    def read_dtype(self, file_name: Path) -> str:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dtype(file_name)


image_reader = ImageReaderManager()

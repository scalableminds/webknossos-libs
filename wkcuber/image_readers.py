from pathlib import Path
from typing import Tuple, Dict, Union, Optional, List

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
        selected_channel: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        pass

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        pass

    def read_channel_count(self, file_name: Path) -> int:
        pass

    def read_sample_count(
        self, file_name: Path  # pylint: disable=unused-argument
    ) -> int:
        return 1

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
        selected_channel: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        this_layer = np.array(Image.open(file_name), dtype)
        this_layer = this_layer.swapaxes(0, 1)
        if selected_channel is not None and this_layer.ndim == 3:
            this_layer = this_layer[:, :, selected_channel[0] : selected_channel[1]]
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
        selected_channel: Optional[Tuple[int, int]],
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
        selected_channel: Optional[Tuple[int, int]],
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
        self.channel_count: Optional[int] = None
        self.z_axis_before_c: Optional[bool] = None
        self.c_count: Optional[int] = None
        self.z_count: Optional[int] = None
        self.x_page_index: Optional[int] = None
        self.y_page_index: Optional[int] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.z_axis_name: Optional[str] = None

    @staticmethod
    def find_correct_channels(
        tif_file: TiffFile,
        selected_channel: Optional[Tuple[int, int]],
    ) -> Dict[int, Optional[Tuple[int, int]]]:
        result: Dict[int, Optional[Tuple[int, int]]] = dict()
        s_count = TiffImageReader.find_count_of_axis(tif_file, "S")
        c_count = TiffImageReader.find_count_of_axis(tif_file, "C")
        if s_count == 1:
            # no s axis
            if c_count == 1:
                result[0] = None
            elif selected_channel is None:
                for i in range(c_count):
                    result[i] = None
            else:
                for i in range(selected_channel[0], selected_channel[1]):
                    result[i] = None
        else:
            # s axis present -> select correct channel + samples per page
            if c_count == 1:
                start_index = selected_channel[0] if selected_channel is not None else 0
                end_index = (
                    selected_channel[1] if selected_channel is not None else s_count
                )
                result[0] = (start_index, end_index)
            elif selected_channel is None:
                for i in range(c_count):
                    result[i] = (0, s_count)
            else:
                # we add one to the outer bound because it is exclusive
                for i in range(
                    selected_channel[0] // s_count,
                    (selected_channel[1] - 1) // s_count + 1,
                ):
                    start_index = selected_channel[0] - i * s_count
                    end_index = selected_channel[1] - i * s_count
                    result[i] = (
                        start_index if start_index > 0 else 0,
                        end_index if end_index < s_count else s_count,
                    )
        return result

    def is_z_axis_before_c(self, tif_file: TiffFile) -> bool:
        assert len(tif_file.series) == 1, "only single tif series are supported"
        tif_series = tif_file.series[0]
        c_index = tif_series.axes.find("C")
        z_index = tif_series.axes.find(self.z_axis_name)
        return c_index == -1 or z_index < c_index

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
        selected_channel: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        with TiffFile(file_name) as tif_file:
            if self.z_axis_name is None:
                self._find_right_z_axis_name(tif_file)
            if self.channel_count is None:
                self.channel_count = self.read_channel_count(file_name)
            if self.z_axis_before_c is None:
                self.z_axis_before_c = self.is_z_axis_before_c(tif_file)
            if self.c_count is None:
                self.c_count = TiffImageReader.find_count_of_axis(tif_file, "C")
            if self.z_count is None:
                assert self.z_axis_name is not None, "Z axis name unclear"
                self.z_count = TiffImageReader.find_count_of_axis(
                    tif_file, self.z_axis_name
                )
            if self.x_page_index is None:
                self.x_page_index = tif_file.pages[0].axes.find("X")
            if self.y_page_index is None:
                self.y_page_index = tif_file.pages[0].axes.find("Y")
            if self.width is None or self.height is None:
                self.width, self.height = self.read_dimensions(file_name)

            output_channel = (
                selected_channel[1] - selected_channel[0]
                if selected_channel is not None
                else self.channel_count
            )
            output_shape = (output_channel, self.width, self.height)

            output = np.empty(output_shape, tif_file.pages[0].dtype)

            channel_to_samples = TiffImageReader.find_correct_channels(
                tif_file, selected_channel
            )

            z_index = z_slice if len(tif_file.pages) > self.c_count else 0

            output_channel_offset = 0
            for channel, sample_slice in channel_to_samples.items():
                # pylint: disable=cell-var-from-loop
                next_channel_offset = (
                    (output_channel_offset + sample_slice[1] - sample_slice[0])
                    if sample_slice is not None
                    else output_channel_offset + 1
                )
                output[output_channel_offset:next_channel_offset] = np.array(
                    list(
                        map(
                            lambda x: x.transpose(
                                (2, self.x_page_index, self.y_page_index)
                            ),
                            map(
                                #  this should be safe to ignore according to https://stackoverflow.com/questions/25314547/cell-var-from-loop-warning-from-pylint since we are directly executing the lambda
                                lambda x: x[
                                    :,
                                    :,
                                    sample_slice[0] : sample_slice[1],
                                ]
                                if sample_slice is not None
                                else x[..., np.newaxis],
                                map(
                                    lambda x: x.asarray(),
                                    tif_file.pages[
                                        z_index * self.c_count
                                        + channel : z_index * self.c_count
                                        + channel
                                        + 1
                                    ]
                                    if self.z_axis_before_c
                                    else tif_file.pages[
                                        channel * self.z_count
                                        + z_index : channel * self.z_count
                                        + z_index
                                    ],
                                ),
                            ),
                        )
                    ),
                    dtype,
                )
                output_channel_offset = next_channel_offset

            output = output.transpose((1, 2, 0))

            output = output.reshape(output.shape + (1,))
            return output

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        with TiffFile(file_name) as tif_file:
            return (
                TiffImageReader.find_count_of_axis(tif_file, "X"),
                TiffImageReader.find_count_of_axis(tif_file, "Y"),
            )

    def read_channel_count(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            return TiffImageReader.find_count_of_axis(
                tif_file, "C"
            ) * TiffImageReader.find_count_of_axis(tif_file, "S")

    def read_sample_count(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            return TiffImageReader.find_count_of_axis(tif_file, "S")

    def read_z_slices_per_file(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            if self.z_axis_name is None:
                self._find_right_z_axis_name(tif_file)
            assert self.z_axis_name is not None, "Z axis name still unclear"
            return TiffImageReader.find_count_of_axis(tif_file, self.z_axis_name)

    def read_dtype(self, file_name: Path) -> str:
        with TiffFile(file_name) as tif_file:
            return tif_file.series[  # pylint: disable=unsubscriptable-object
                0
            ].dtype.name

    def _find_right_z_axis_name(self, tif_file: TiffFile) -> None:
        i_count = TiffImageReader.find_count_of_axis(tif_file, "I")
        z_count = TiffImageReader.find_count_of_axis(tif_file, "Z")
        q_count = TiffImageReader.find_count_of_axis(tif_file, "Q")
        if i_count > 1:
            assert (
                z_count * q_count == 1
            ), "Format error, as multiple Z axis names were identified"
            self.z_axis_name = "I"
        elif q_count > 1:
            assert (
                z_count * i_count == 1
            ), "Format error, as multiple Z axis names were identified"
            self.z_axis_name = "Q"
        else:
            assert (
                i_count * q_count == 1
            ), "Format error, as multiple Z axis names were identified"
            self.z_axis_name = "Z"


class CziImageReader(ImageReader):
    def __init__(self) -> None:
        self.tile_shape: Optional[List[int]] = None
        self.channel_count: Optional[int] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.existing_axes: Optional[List[int]] = None

    def matches_if_exist(self, index: int, value: int, other_value: int) -> bool:
        assert self.existing_axes is not None, "Axes initialization failed"
        if self.existing_axes[index] == 1:
            return value == other_value
        else:
            return True

    def get_access_index_from_existence_array(self, wanted_index: int) -> int:
        assert self.existing_axes is not None, "Axes initialization failed"
        index_sum = 0
        for i in range(wanted_index):
            index_sum += self.existing_axes[i]
        return index_sum

    @staticmethod
    def find_count_of_axis(czi_file: CziFile, axis: str) -> int:
        index = czi_file.axes.find(axis)
        if index == -1:
            return 1
        else:
            return czi_file.shape[index]

    def select_correct_tiles(
        self,
        selected_channel: Optional[Tuple[int, int]],
        tile_shape: List[int],
        dataset_shape: List[int],
    ) -> Dict[int, Dict[Tuple[int, int], Dict[int, Tuple[int, int]]]]:
        assert self.existing_axes is not None, "Axes initialization failed"
        # return type is C number to C slice to 0 number to 0 slice
        result = dict()
        # C axis exists
        c_count_per_tile = 1
        c_tiles_for_complete_axis = 1
        if self.existing_axes[1] == 1:
            c_index = self.get_access_index_from_existence_array(1)
            c_count_per_tile = tile_shape[c_index]
            c_tiles_for_complete_axis = dataset_shape[c_index] // tile_shape[c_index]
        # 0 axis exists always
        zero_index = self.get_access_index_from_existence_array(4)
        zero_count_per_tile = tile_shape[zero_index]
        zero_tiles_for_complete_axis = (
            dataset_shape[zero_index] // tile_shape[zero_index]
        )

        if selected_channel is None:
            if self.existing_axes[1] == 1:
                for i in range(c_tiles_for_complete_axis):
                    for j in range(zero_tiles_for_complete_axis):
                        result[i] = {
                            (0, c_count_per_tile): {j: (0, zero_count_per_tile)}
                        }
            else:
                for i in range(zero_tiles_for_complete_axis):
                    result[0] = {(0, 1): {i: (0, zero_count_per_tile)}}
        else:
            if self.existing_axes[1] == 1:
                for i in range(
                    selected_channel[0] // (c_count_per_tile * zero_count_per_tile),
                    (selected_channel[1] - 1)
                    // (c_count_per_tile * zero_count_per_tile)
                    + 1,
                ):
                    # TODO come up with the formula to replace the loops
                    c_range = [None, 0]
                    zero_map = dict()
                    for j in range(c_count_per_tile):
                        zero_range = [None, 0]
                        for k in range(zero_count_per_tile):
                            index = (
                                i * (c_count_per_tile * zero_count_per_tile)
                                + j * zero_count_per_tile
                                + k
                            )
                            if selected_channel[0] <= index < selected_channel[1]:
                                if c_range[0] is None:
                                    c_range[0] = j
                                elif c_range[1] is not None and c_range[1] < j:
                                    c_range[1] = j
                                if zero_range[0] is None:
                                    zero_range[0] = k
                                elif zero_range[1] is not None and zero_range[1] < k:
                                    zero_range[1] = k
                        assert (
                            zero_range[0] is not None and zero_range[1] is not None
                        ), "Could not create correct tile format"
                        zero_map[j] = (zero_range[0], zero_range[1] + 1)
                    assert (
                        c_range[0] is not None and c_range[1] is not None
                    ), "Could not create correct tile format"
                    result[i] = {(c_range[0], c_range[1] + 1): zero_map}
            else:
                zero_map = dict()
                for i in range(
                    selected_channel[0] // zero_count_per_tile,
                    (selected_channel[1] - 1) // zero_count_per_tile + 1,
                ):
                    start_index = selected_channel[0] - i * zero_count_per_tile
                    end_index = selected_channel[1] - i * zero_count_per_tile
                    zero_map[i] = (
                        start_index if start_index > 0 else 0,
                        end_index
                        if end_index < zero_count_per_tile
                        else zero_count_per_tile,
                    )
                result[0] = {(0, 1): zero_map}

        return result

    def read_array(
        self,
        file_name: Path,
        dtype: np.dtype,
        z_slice: int,
        selected_channel: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        with CziFile(file_name) as czi_file:
            # pylint: disable=unsubscriptable-object
            if self.channel_count is None:
                self.channel_count = self.read_channel_count(file_name)
            if self.width is None or self.height is None:
                self.width, self.height = self.read_dimensions(file_name)
            # we assume the tile shape is constant, so we can cache it
            if self.tile_shape is None:
                self.tile_shape = czi_file.filtered_subblock_directory[0].shape
            if self.existing_axes is None:
                # according to specification, axes order is always ZCYX0
                possible_axes = ["Z", "C", "Y", "X", "0"]
                self.existing_axes = [0, 0, 0, 0, 0]
                for i, axis in enumerate(possible_axes):
                    if czi_file.axes.find(axis) != -1:  # pylint: disable=no-member
                        self.existing_axes[i] = 1

            assert self.tile_shape is not None, "Cannot read tile shape format."

            output_channel = (
                selected_channel[1] - selected_channel[0]
                if selected_channel is not None
                else self.channel_count
            )
            output_shape = (output_channel, self.width, self.height)
            output = np.empty(output_shape, dtype)

            z_file_start = czi_file.start[0] if self.existing_axes[0] == 1 else 0
            c_axis_index = self.get_access_index_from_existence_array(1)
            c_file_start = (
                czi_file.start[c_axis_index] if self.existing_axes[1] == 1 else 0
            )
            zero_axis_index = self.get_access_index_from_existence_array(4)
            zero_file_start = czi_file.start[zero_axis_index]
            output_channel_offset = 0
            # Dict[int, Dict[Tuple[int, int], Dict[int, Tuple[int, int]]]]
            for c_index, c_dict in self.select_correct_tiles(
                selected_channel, self.tile_shape, czi_file.shape
            ).items():
                for c_slice, zero_dict in c_dict.items():
                    for zero_index, zero_slice in zero_dict.items():
                        for (
                            entry
                        ) in (
                            czi_file.filtered_subblock_directory  # pylint: disable=not-an-iterable
                        ):
                            if (
                                self.matches_if_exist(
                                    0, entry.start[0] - z_file_start, z_slice
                                )
                                and self.matches_if_exist(
                                    1,
                                    (entry.start[c_axis_index] - c_file_start)
                                    // self.tile_shape[c_axis_index],
                                    c_index,
                                )
                                and self.matches_if_exist(
                                    4,
                                    (entry.start[zero_axis_index] - zero_file_start)
                                    // self.tile_shape[zero_axis_index],
                                    zero_index,
                                )
                            ):
                                data = entry.data_segment().data()
                                data = to_target_datatype(data, dtype)
                                for i in range(c_slice[0], c_slice[1]):
                                    # format of curr data can be (Z=1),Y,X,0, but should be 0,X,Y
                                    curr_data = (
                                        np.take(data, i, c_axis_index)
                                        if self.existing_axes[1] == 1
                                        else data
                                    )
                                    if curr_data.ndim == 5:
                                        curr_data = curr_data.transpose(
                                            (0, 1, -1, -2, -3)
                                        )
                                    elif curr_data.ndim == 4:
                                        curr_data = curr_data.transpose((0, -1, -2, -3))
                                    else:
                                        curr_data = curr_data.transpose((-1, -2, -3))
                                    curr_data = curr_data.reshape(
                                        (
                                            curr_data.shape[-3],
                                            curr_data.shape[-2],
                                            curr_data.shape[-1],
                                        )
                                    )
                                    output[
                                        output_channel_offset : output_channel_offset
                                        + zero_slice[1]
                                        - zero_slice[0]
                                    ] = curr_data
                                    output_channel_offset += (
                                        zero_slice[1] - zero_slice[0] + 1
                                    )

            # CZI stores pixels as BGR instead of RGB, so swap axes to ensure right color output
            if (
                output_channel == 3
                and dtype == np.uint8
                and czi_file.filtered_subblock_directory[0].pixel_type
                == "Bgr24"  # pylint: disable=unsubscriptable-object
            ):
                output[2, :, :], output[0, :, :] = output[0, :, :], output[2, :, :]

            output = output.transpose((1, 2, 0))
            output = output.reshape(output.shape + (1,))

            return output

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        with CziFile(file_name) as czi_file:
            return (
                CziImageReader.find_count_of_axis(czi_file, "X"),
                CziImageReader.find_count_of_axis(czi_file, "Y"),
            )

    def read_channel_count(self, file_name: Path) -> int:
        with CziFile(file_name) as czi_file:
            c_count = CziImageReader.find_count_of_axis(czi_file, "C")
            s_count = CziImageReader.find_count_of_axis(czi_file, "0")
            return c_count * s_count

    def read_sample_count(self, file_name: Path) -> int:
        with CziFile(file_name) as czi_file:
            return CziImageReader.find_count_of_axis(czi_file, "0")

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
        selected_channels: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        _, ext = path.splitext(file_name)

        # Image shape will be (x, y, channel_count, z=1) or (x, y, z=1)
        image = self.readers[ext].read_array(
            file_name, dtype, z_slice, selected_channels
        )
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

    def read_sample_count(self, file_name: Path) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_sample_count(file_name)

    def read_z_slices_per_file(self, file_name: Path) -> int:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_z_slices_per_file(file_name)

    def read_dtype(self, file_name: Path) -> str:
        _, ext = path.splitext(file_name)
        return self.readers[ext].read_dtype(file_name)


# refresh all cached values
def new_image_reader() -> None:
    global image_reader
    image_reader = ImageReaderManager()


image_reader = ImageReaderManager()

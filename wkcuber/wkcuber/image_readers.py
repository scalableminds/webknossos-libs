from pathlib import Path
from typing import Tuple, Dict, Union, Optional

import numpy as np
import logging
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
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
    ) -> np.ndarray:
        pass

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        pass

    def read_channel_count(self, file_name: Path) -> int:
        pass

    # the sample count describes how many values are saved for each pixel, e.g. uint24 is 3x uint8 channels, therefore the sample count is 3. This applies only on TIFF and CZI datasets.
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
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
    ) -> np.ndarray:
        this_layer = np.array(Image.open(file_name), dtype)
        this_layer = this_layer.swapaxes(0, 1)
        if channel_index is not None and this_layer.ndim == 3:
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


def to_target_datatype(
    data: np.ndarray, target_dtype: Union[type, np.dtype]
) -> np.ndarray:
    factor = (1 + np.iinfo(data.dtype).max) / (1 + np.iinfo(target_dtype).max)
    return (data / factor).astype(target_dtype)


class Dm3ImageReader(ImageReader):
    def read_array(
        self,
        file_name: Path,
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
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
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
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
    @staticmethod
    def _get_page_data(tif_file: TiffFile, page_index: int) -> np.ndarray:
        if 0 <= page_index < len(tif_file.pages):
            return tif_file.pages[page_index].asarray()
        elif tif_file.is_imagej:
            # ImageJ can export 3-dimensional, single-page tiff files.
            # In that case, we manually read from the correct filehandle
            # by seeking to the correct position.
            shape = tif_file.pages[0].shape
            elements_per_page = 1
            for i in shape:
                elements_per_page *= i
            series = tif_file.series[0]
            tif_file.filehandle.seek(
                series.offset
                + page_index * np.dtype(series.dtype).itemsize * elements_per_page
            )
            data = tif_file.filehandle.read_array(series.dtype, elements_per_page)
            data = data.reshape(shape)
            return data
        else:
            raise Exception("Cannot read page data!")

    @staticmethod
    def _get_page_index(
        tif_file: TiffFile,
        z_index: int,
        c_index: int,
        s_index: int,
        axes: Dict[str, Tuple[bool, int, int]],
        z_axis_name: str,
        page_axes: Dict[str, Tuple[int, int, int]],
    ) -> int:
        # TiffFile consists of Tiff Pages, which are ordered according to the axis order
        # To find the index of the page that contains the data we want, we increase the page_index for each axis that we are looking for
        # Additionally, we multiply the index with dimension of each axis to "skip" the necessary amount of pages.
        # We divide by an axis if it is present in the TiffPage because there are less TiffPages to skip.
        page_index = 0
        for axis in tif_file.series[0].axes:
            page_index *= axes[axis][2]
            if axis in page_axes:
                page_index //= page_axes[axis][2]
            if axis == z_axis_name:
                page_index += z_index
            elif axis == "C":
                page_index += c_index
            elif axis == "S":
                page_index += s_index
        return page_index

    @staticmethod
    def _find_correct_pages(
        tif_file: TiffFile,
        channel_index: Optional[int],
        sample_index: Optional[int],
        z_index: int,
        axes: Dict[str, Tuple[bool, int, int]],
        z_axis_name: str,
        page_axes: Dict[str, Tuple[int, int, int]],
    ) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
        # return format is page_index to [Optional[channel_index], Optional[sample_index]]
        result: Dict[int, Tuple[Optional[int], Optional[int]]] = dict()

        if channel_index is not None:
            page_c_index = (
                (channel_index // page_axes["C"][2])
                if "C" in page_axes
                else channel_index
            )
            data_c_index = (
                (channel_index % page_axes["C"][2]) if "C" in page_axes else None
            )
            page_s_index = 0
            data_s_index = None
            if sample_index is not None:
                page_s_index = (
                    (sample_index // page_axes["S"][2])
                    if "S" in page_axes
                    else sample_index
                )
                data_s_index = (
                    (sample_index % page_axes["S"][2]) if "S" in page_axes else None
                )

            result[
                TiffImageReader._get_page_index(
                    tif_file,
                    z_index,
                    page_c_index,
                    page_s_index,
                    axes,
                    z_axis_name,
                    page_axes,
                )
            ] = (data_c_index, data_s_index)
        else:
            # implies sample_index not set
            c_range = range(1)
            s_range = range(1)
            if "C" in page_axes and axes["C"][0]:
                c_range = range(axes["C"][2] // page_axes["C"][2])
            elif axes["C"][0]:
                c_range = range(axes["C"][2])

            if "S" in page_axes and axes["S"][0]:
                s_range = range(axes["S"][2] // page_axes["S"][2])
            elif axes["S"][0]:
                s_range = range(axes["S"][2])

            for c in c_range:
                for s in s_range:
                    result[
                        TiffImageReader._get_page_index(
                            tif_file, z_index, c, s, axes, z_axis_name, page_axes
                        )
                    ] = (None, None)

        return result

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
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
    ) -> np.ndarray:
        with TiffFile(file_name) as tif_file:
            # Read metadata
            z_axis_name = self._find_right_z_axis_name(tif_file)
            used_axes = {"C", "Y", "X", "S"} | {z_axis_name}
            assert len(tif_file.series) == 1, "Multi-series tiff not supported"
            series = tif_file.series[0]  # pylint: disable=unsubscriptable-object
            page = tif_file.pages[0]
            axes = {}
            for index, axis in enumerate(series.axes):
                axes[axis] = (True, index, series.shape[index])
            for axis in used_axes:
                if axis not in axes:
                    axes[axis] = (False, 0, 1)
            count = 0
            page_axes = {}
            for index, axis in enumerate(page.axes):
                # when reading the data, we'll remove any axis that is not used, so the dimensionality shrinks
                page_axes[axis] = (index, index - count, page.shape[index])
                if axis not in used_axes:
                    count += 1
            assert (
                len(axes) > 0 and z_axis_name is not None
            ), "TiffReader initialization failed"
            # this calculation is possible this way because the extent for non existing axes gets set to one
            num_output_channel = axes["C"][2] * axes["S"][2]
            if channel_index is not None:
                num_output_channel //= axes["C"][2]
            if sample_index is not None:
                num_output_channel //= axes["S"][2]

            output_shape = (axes["X"][2], axes["Y"][2], num_output_channel)
            output = np.empty(output_shape, tif_file.pages[0].dtype)

            z_index = z_slice if axes[z_axis_name][0] else 0
            output_channel_offset = 0

            # Read the data
            for page_index, (data_c_index, data_s_index,) in self._find_correct_pages(
                tif_file,
                channel_index,
                sample_index,
                z_index,
                axes,
                z_axis_name,
                page_axes,
            ).items():
                page_data = self._get_page_data(tif_file, page_index)
                # remove any axis that we do not use
                # left over axes [(C)YX(S)] (Z axis is not on same TiffPage)
                for axis in page_axes:
                    if axis not in used_axes:
                        page_data = page_data.take([0], page_axes[axis][1])

                # axes order is then [YX(S)(C)]
                if page_data.ndim == 4:
                    page_data = page_data.transpose(
                        (
                            page_axes["X"][1],
                            page_axes["Y"][1],
                            page_axes["S"][1],
                            page_axes["C"][1],
                        )
                    )
                elif page_data.ndim == 3:
                    if "S" in page_axes:
                        page_data = page_data.transpose(
                            (
                                page_axes["X"][1],
                                page_axes["Y"][1],
                                page_axes["S"][1],
                            )
                        )
                    else:
                        page_data = page_data.transpose(
                            (
                                page_axes["X"][1],
                                page_axes["Y"][1],
                                page_axes["C"][1],
                            )
                        )
                elif page_data.ndim == 2:
                    page_data = page_data.transpose(
                        (page_axes["X"][1], page_axes["Y"][1])
                    )
                else:
                    raise Exception

                if data_c_index is not None:
                    page_data = page_data.take([data_c_index], -1)
                if data_s_index is not None:
                    page_data = page_data.take([data_s_index], -1)

                if page_data.ndim == 4:
                    # means C and S axis present and no data index set
                    for c in range(page_data.shape[-1]):
                        output[
                            :,
                            :,
                            output_channel_offset : output_channel_offset
                            + page_axes["S"][2],
                        ] = page_data[:, :, :, c]
                        output_channel_offset += (
                            output_channel_offset + page_axes["S"][2]
                        )
                elif page_data.ndim == 3:
                    next_channel_offset = output_channel_offset + page_data.shape[-1]
                    output[:, :, output_channel_offset:next_channel_offset] = page_data
                    output_channel_offset = next_channel_offset
                elif page_data.ndim == 2:
                    output[:, :, output_channel_offset] = page_data
                    output_channel_offset += 1
                else:
                    raise Exception("Invalid axis count")

            output = np.array(output, dtype)
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
            return TiffImageReader.find_count_of_axis(tif_file, "C")

    def read_sample_count(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            return TiffImageReader.find_count_of_axis(tif_file, "S")

    def read_z_slices_per_file(self, file_name: Path) -> int:
        with TiffFile(file_name) as tif_file:
            z_axis_name = self._find_right_z_axis_name(tif_file)
            return TiffImageReader.find_count_of_axis(tif_file, z_axis_name)

    def read_dtype(self, file_name: Path) -> str:
        with TiffFile(file_name) as tif_file:
            return tif_file.series[  # pylint: disable=unsubscriptable-object
                0
            ].dtype.name

    def _find_right_z_axis_name(self, tif_file: TiffFile) -> str:
        i_count = TiffImageReader.find_count_of_axis(tif_file, "I")
        z_count = TiffImageReader.find_count_of_axis(tif_file, "Z")
        q_count = TiffImageReader.find_count_of_axis(tif_file, "Q")
        if i_count > 1:
            assert (
                z_count * q_count == 1
            ), "Format error, as multiple Z axis names were identified"
            return "I"
        elif q_count > 1:
            assert (
                z_count * i_count == 1
            ), "Format error, as multiple Z axis names were identified"
            return "Q"
        else:
            assert (
                i_count * q_count == 1
            ), "Format error, as multiple Z axis names were identified"
            return "Z"


class CziImageReader(ImageReader):
    @staticmethod
    def _select_correct_tiles_for_czi(
        channel_index: Optional[int],
        sample_index: Optional[int],
        z_slice: int,
        axes: Dict[str, Tuple[bool, int, int]],
        tile_shape: Tuple[int, int],
        dataset_shape: Tuple[int, ...],
    ) -> Dict[int, Tuple[int, Optional[int], Optional[int], Optional[int]]]:
        assert len(axes) != 0, "CZIReader initialization failed"
        # return type is tile channel index to tile z index, data z index, data channel index and 0 index, None means all
        result: Dict[
            int, Tuple[int, Optional[int], Optional[int], Optional[int]]
        ] = dict()
        # C axis exists
        c_count_per_tile = 1
        c_tiles_for_complete_axis = 1
        if axes["C"][0]:
            c_count_per_tile = tile_shape[axes["C"][1]]
            c_tiles_for_complete_axis = dataset_shape[axes["C"][1]] // c_count_per_tile

        tile_z_index: int = 0
        data_z_index: Optional[int] = None
        if axes["Z"][0]:
            tile_z_index = z_slice // dataset_shape[axes["Z"][1]]
            data_z_index = z_slice % dataset_shape[axes["Z"][1]]

        if channel_index is not None:
            tile_channel_index = channel_index // c_count_per_tile
            data_channel_index = channel_index % c_count_per_tile
            result[tile_channel_index] = (
                tile_z_index,
                data_z_index,
                data_channel_index,
                sample_index,
            )
        else:
            # no channel index set, implies no sample index set
            for i in range(c_tiles_for_complete_axis):
                result[i] = (tile_z_index, data_z_index, None, None)

        return result

    @staticmethod
    def _matches_if_exist(
        axis: str, value: int, other_value: int, axes: Dict[str, Tuple[bool, int, int]]
    ) -> bool:
        assert len(axes) != 0, "Axes initialization failed"
        if axes[axis][0]:
            return value == other_value
        else:
            return True

    @staticmethod
    def find_count_of_axis(czi_file: CziFile, axis: str) -> int:
        index = czi_file.axes.find(axis)
        if index == -1:
            return 1
        else:
            return czi_file.shape[index]

    def read_array(
        self,
        file_name: Path,
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int],
        sample_index: Optional[int],
    ) -> np.ndarray:
        with CziFile(file_name) as czi_file:
            # Read metadata
            tile_shape = czi_file.filtered_subblock_directory[  # pylint: disable=unsubscriptable-object
                0
            ].shape
            dataset_shape: Tuple[int, ...] = tuple(czi_file.shape)
            count = 0
            used_axes = {"Z", "C", "Y", "X", "0"}
            axes: Dict[str, Tuple[bool, int, int]] = {}
            for i, axis in enumerate(czi_file.axes):
                axes[axis] = (True, i, i - count)
                if axis not in used_axes:
                    count += 1
            for axis in used_axes:
                if axis not in axes:
                    axes[axis] = (False, 0, 0)

            assert tile_shape is not None, "Cannot read tile shape format."

            if sample_index is not None:
                num_output_channel = 1
            elif channel_index is not None:
                num_output_channel = dataset_shape[axes["0"][1]]
            else:
                channel_count = dataset_shape[axes["C"][1]] if axes["C"][0] else 1
                num_output_channel = channel_count * dataset_shape[axes["0"][1]]

            output_shape = (
                dataset_shape[axes["X"][1]],
                dataset_shape[axes["Y"][1]],
                num_output_channel,
            )
            output = np.empty(output_shape, dtype)

            z_file_start = (
                czi_file.start[axes["Z"][1]]  # pylint: disable=unsubscriptable-object
                if axes["Z"][0]
                else 0
            )
            c_file_start = (
                czi_file.start[axes["C"][1]]  # pylint: disable=unsubscriptable-object
                if axes["C"][0]
                else 0
            )
            output_channel_offset = 0

            # Read the data
            for tile_channel_index, (
                tile_z_index,
                data_z_index,
                data_channel_index,
                data_sample_index,
            ) in CziImageReader._select_correct_tiles_for_czi(
                channel_index, sample_index, z_slice, axes, tile_shape, dataset_shape
            ).items():
                # since the czi tiles are not sorted, we search linearly through them and check if the tile matches with the wanted coordinates
                # however, some axes might not exist, so the _matches_if_exist helper returns true if the axis does not exist
                for (
                    entry
                ) in (
                    czi_file.filtered_subblock_directory  # pylint: disable=not-an-iterable
                ):
                    if CziImageReader._matches_if_exist(
                        "Z",
                        entry.start[axes["Z"][1]] - z_file_start,
                        tile_z_index,
                        axes,
                    ) and CziImageReader._matches_if_exist(
                        "C",
                        (entry.start[axes["C"][1]] - c_file_start)
                        // tile_shape[axes["C"][1]],
                        tile_channel_index,
                        axes,
                    ):
                        data = entry.data_segment().data()
                        data = to_target_datatype(data, dtype)

                        # left axis are in order [(Z)(C)0XY]
                        for axis in czi_file.axes:  # pylint: disable=not-an-iterable
                            if axis not in used_axes:
                                data = data.take([0], axes[axis][2])
                        if data.ndim == 5:
                            data = data.transpose(
                                (
                                    axes["Z"][2],
                                    axes["X"][2],
                                    axes["Y"][2],
                                    axes["0"][2],
                                    axes["C"][2],
                                )
                            )
                        elif data.ndim == 4:
                            if axes["Z"][0]:
                                data = data.transpose(
                                    (
                                        axes["Z"][2],
                                        axes["X"][2],
                                        axes["Y"][2],
                                        axes["0"][2],
                                    )
                                )
                            else:
                                data = data.transpose(
                                    (
                                        axes["X"][2],
                                        axes["Y"][2],
                                        axes["0"][2],
                                        axes["C"][2],
                                    )
                                )
                        elif data.ndim == 3:
                            data = data.transpose(
                                (
                                    axes["X"][2],
                                    axes["Y"][2],
                                    axes["0"][2],
                                )
                            )
                        else:
                            raise Exception

                        if data_z_index is not None:
                            data = data.take([data_z_index], 0)

                        # if z axis exist, data_z_index is set, so the access index for this is always 0
                        if data_channel_index is not None and axes["C"][0]:
                            data = data.take([data_channel_index], -1)

                        # if the sample index is set, the channel index is set, too, so access index is always 0
                        if data_sample_index is not None:
                            data = data.take([data_sample_index], -1)

                        # special case with C and 0 and no selected index set
                        if data.ndim == 4:
                            for c in range(tile_shape[axes["C"][1]]):
                                next_output_channel_offset = (
                                    output_channel_offset + data.shape[2]
                                )
                                output[
                                    :,
                                    :,
                                    output_channel_offset:next_output_channel_offset,
                                ] = data[:, :, :, c]
                                output_channel_offset = next_output_channel_offset
                        elif data.ndim == 2:
                            output[:, :, output_channel_offset] = data
                            output_channel_offset += 1
                        else:
                            output_channel_increment = (
                                1 if data_sample_index is not None else data.shape[-1]
                            )
                            next_output_channel_offset = (
                                output_channel_offset + output_channel_increment
                            )
                            output[
                                :, :, output_channel_offset:next_output_channel_offset
                            ] = data
                            output_channel_offset = next_output_channel_offset

        # CZI stores pixels as BGR instead of RGB, so swap axes to ensure right color output
        if (
            num_output_channel == 3
            and dtype == np.uint8
            and czi_file.filtered_subblock_directory[  # pylint: disable=unsubscriptable-object
                0
            ].pixel_type
            == "Bgr24"
        ):
            output[:, :, 2], output[:, :, 0] = output[:, :, 0], output[:, :, 2]

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
            return CziImageReader.find_count_of_axis(czi_file, "C")

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
            # TIFF file endings
            ".tif": TiffImageReader(),
            ".tiff": TiffImageReader(),
            # JPEG file endings
            ".jpg": PillowImageReader(),
            ".jpeg": PillowImageReader(),
            ".jpe": PillowImageReader(),
            ".jfif": PillowImageReader(),
            ".jif": PillowImageReader(),
            # JPEG2000 file endings
            ".jp2": PillowImageReader(),
            ".j2k": PillowImageReader(),
            ".jpf": PillowImageReader(),
            ".jpm": PillowImageReader(),
            ".jpg2": PillowImageReader(),
            ".j2c": PillowImageReader(),
            ".jpc": PillowImageReader(),
            ".jpx": PillowImageReader(),
            ".mj2": PillowImageReader(),
            # Other image file endings
            ".png": PillowImageReader(),
            ".dm3": Dm3ImageReader(),
            ".dm4": Dm4ImageReader(),
            ".czi": CziImageReader(),
        }

    def read_array(
        self,
        file_name: Path,
        dtype: Union[type, np.dtype],
        z_slice: int,
        channel_index: Optional[int] = None,
        sample_index: Optional[int] = None,
    ) -> np.ndarray:
        ext = file_name.suffix

        # Image shape will be (x, y, channel_count, z=1) or (x, y, z=1)
        image = self.readers[ext].read_array(
            file_name, dtype, z_slice, channel_index, sample_index
        )
        # Standardize the image shape to (x, y, channel_count, z=1)
        if image.ndim == 3:
            image = image.reshape(image.shape + (1,))

        return image

    def read_dimensions(self, file_name: Path) -> Tuple[int, int]:
        ext = file_name.suffix
        return self.readers[ext].read_dimensions(file_name)

    def read_channel_count(self, file_name: Path) -> int:
        ext = file_name.suffix
        return self.readers[ext].read_channel_count(file_name)

    def read_sample_count(self, file_name: Path) -> int:
        ext = file_name.suffix
        return self.readers[ext].read_sample_count(file_name)

    def read_z_slices_per_file(self, file_name: Path) -> int:
        ext = file_name.suffix
        return self.readers[ext].read_z_slices_per_file(file_name)

    def read_dtype(self, file_name: Path) -> str:
        ext = file_name.suffix
        return self.readers[ext].read_dtype(file_name)


image_reader = ImageReaderManager()

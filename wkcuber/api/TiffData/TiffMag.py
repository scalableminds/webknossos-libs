from typing import Optional, List, Generator

from skimage import io
import numpy as np
import os
from re import findall
from glob import iglob
from itertools import zip_longest


def replace_coordinate(pattern: str, coord_id: str, coord: int) -> str:
    occurrences = findall("{" + coord_id + "+}", pattern)
    for occurrence in occurrences:
        number_of_digits = len(occurrence) - 2
        if number_of_digits > 1:
            format_str = "0" + str(number_of_digits) + "d"
        else:
            format_str = "d"
        pattern = pattern.replace(occurrence, format(coord, format_str), 1)
    return pattern


def to_file_name(z) -> str:
    return replace_coordinate("{zzzzz}.tif", "z", z)


def detect_value(
    pattern_element: str,
    ls_item: str,
    dim: str,
    ignore_dims: Optional[List[str]] = None,
) -> List[int]:
    if ignore_dims is not None:
        for ignore_dim in ignore_dims:
            pattern_element = pattern_element.replace("{" + ignore_dim, ignore_dim)
            pattern_element = pattern_element.replace(ignore_dim + "}", ignore_dim)

    if "{" + dim in pattern_element and dim + "}" in pattern_element:
        open_position = pattern_element.find("{" + dim)
        close_position = pattern_element.find(dim + "}")
        try:
            substring = ls_item[open_position:close_position]
            return [int(substring)]
        except ValueError:
            raise ValueError(
                f"Failed to autodetect tile ranges, there were files not matching the pattern: {ls_item} does not match {pattern_element}"
            )
    return []


class TiffMag:
    def __init__(self, root, header):
        x_range = [0]  # currently tiled tiffs are not supported
        y_range = [0]  # currently tiled tiffs are not supported

        self.root = root
        self.tiffs = dict()
        self.dtype = header.dtype
        self.num_channels = header.num_channels

        pattern = "{zzzzz}.tif"

        z_range = [
            detect_value(pattern, file_name, dim="z")[0]
            for file_name in self.list_files()
        ]

        for z in z_range:
            self.tiffs[z] = TiffReader.open(
                self.get_file_name_for_layer(z)
            )  # open is lazy

    def read(self, off, shape) -> np.array:
        if not self.has_only_one_channel():
            # modify the shape to also include the num_channels
            shape = tuple(shape) + tuple([self.num_channels])

        data = np.zeros(shape=shape, dtype=self.dtype)
        for i, (z, offset, size) in enumerate(
            self.calculate_relevant_slices(off, shape)
        ):
            if z in self.tiffs:
                data[:, :, i] = np.array(self.tiffs[z].read(), self.dtype)[
                    offset[0] : offset[0] + size[0], offset[1] : offset[1] + size[1]
                ]
            else:
                shape_without_z = shape[:2] + shape[3:]
                data[:, :, i] = np.zeros(shape_without_z, self.dtype)

        if self.has_only_one_channel():
            # convert data into shape with dedicated num_channels (len(data.shape) == 4)
            # this only effects data where the num_channel is 1 and therefore len(data.shape) was 3
            # this makes it easier to handle both, multi-channel and single-channel, similar
            data = np.expand_dims(data, 3)

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, -1, 0)
        return data

    def write(self, off, data):
        # convert data into shape with dedicated num_channels (len(data.shape) == 4)
        # this only effects data where the num_channel is 1 and therefore len(data.shape) was 3
        # this makes it easier to handle both, multi-channel and single-channel, similar
        data = data.reshape((-1,) + data.shape[-3:])

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, 0, -1)

        self.assert_correct_data_format(data)

        for i, (z, offset, _) in enumerate(
            self.calculate_relevant_slices(off, data.shape)
        ):
            # initialize images for z_layers that did not exist before
            if z not in self.tiffs:
                total_shape = [
                    sum(x)
                    for x in zip_longest(data[:, :, i].shape, offset, fillvalue=0)
                ]
                if self.has_only_one_channel():
                    # Convert single-channel data into the expected format
                    # E.g. convert shape (300, 300, 1) into (300, 300)
                    total_shape = tuple(total_shape)[:-1]

                self.tiffs[z] = TiffReader.init_tiff(
                    np.zeros(total_shape, self.dtype), self.get_file_name_for_layer(z)
                )

            # write new pixel data into the image
            pixel_data = (
                data[:, :, i] if not self.has_only_one_channel() else data[:, :, i, 0]
            )

            self.tiffs[z].merge_with_image(pixel_data, offset)

    def compress(self, dst_path: str, compress_files: bool = False):
        raise NotImplementedError

    def list_files(self):
        file_paths = list(iglob(os.path.join(self.root, "*.tif")))

        for file_path in file_paths:
            yield os.path.relpath(os.path.normpath(file_path), self.root)

    def close(self):
        return

    def calculate_relevant_slices(self, offset, shape):
        for z in range(offset[2] + 1, offset[2] + shape[2] + 1):
            yield tuple(
                (z, offset[0:2], shape[0:2])
            )  # return tuple of important z layers an the x-y offset (without z offset) and the size (without z length)

    def has_only_one_channel(self) -> bool:
        return self.num_channels == 1

    def assert_correct_data_format(self, data):
        if not len(data.shape) == 4:
            raise AttributeError(
                "The shape of the provided data does not match the expected shape."
            )
        if not data.shape[3] == self.num_channels:
            raise AttributeError(
                f"The shape of the provided data does not match the expected shape. (Expected {self.num_channels} channels)"
            )
        if not np.dtype(data.dtype) == self.dtype:
            raise AttributeError(
                f"The type of the provided data does not match the expected type. (Expected np.array of type {self.dtype.name})"
            )

    def get_file_name_for_layer(self, z) -> str:
        return os.path.join(self.root, to_file_name(z))

    @staticmethod
    def open(root: str, header=None):
        if header is None:
            header = TiffMagHeader()
        return TiffMag(root, header)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


class TiffMagHeader:
    def __init__(self, pattern="{z}.tif", dtype=np.dtype("uint8"), num_channels=1):
        self.pattern = pattern
        self.dtype = np.dtype(dtype)
        self.num_channels = num_channels


class TiffReader:
    def __init__(self, file_name):
        self.file_name = file_name

    @classmethod
    def init_tiff(cls, pixels, file_name):
        tr = TiffReader(file_name)
        tr.write(pixels)
        return tr

    @classmethod
    def open(cls, file_name):
        return cls(file_name)

    def read(self) -> np.array:
        return io.imread(self.file_name)

    def write(self, pixels):
        io.imsave(self.file_name, pixels, check_contrast=False)

    def merge_with_image(self, foreground_pixels, offset):
        background_pixels = self.read()
        bg_shape = background_pixels.shape
        fg_shape = foreground_pixels.shape

        fg_shape_with_off = [sum(x) for x in zip_longest(fg_shape, offset, fillvalue=0)]
        total_shape = [max(x) for x in zip(bg_shape, fg_shape_with_off)]
        new_image = np.zeros(total_shape, dtype=background_pixels.dtype)

        new_image[0 : bg_shape[0], 0 : bg_shape[1]] = background_pixels
        new_image[
            offset[0] : fg_shape_with_off[0], offset[1] : fg_shape_with_off[1]
        ] = foreground_pixels
        self.write(new_image)

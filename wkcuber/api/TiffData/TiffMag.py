from typing import Optional, List

from PIL import Image
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


def to_file_name(z):
    return replace_coordinate("test.000{z}.tiff", "z", z)


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


"""
def read_tiled_slice(
    tiled_dataset_path_parent: Optional[str],
    tiled_dataset_path_pattern: str,
    x_range: range,
    y_range: range,
    z: int,
    initial_read: bool = False,
    z_range: Optional[range] = None,
) -> np.ndarray:
    image_grid = []
    for x in x_range:
        col = []
        for y in y_range:
            real_pos = find_closest_non_skip_tile(
                np.array([z, x, y]), skip_tiles, z_range
            )

            current_path = SliceReader3D.fill_path_pattern(
                tiled_dataset_path_pattern, real_pos
            )
            if tiled_dataset_path_parent is not None:
                current_path = os.path.join(tiled_dataset_path_parent, current_path)

            logger.debug("Reading image {}".format(current_path))

            col.append(
                read_image(current_path, initial_read=initial_read).transpose()
            )
        image_grid.append(col)
    return np.array(image_grid)
"""


class TiffMag:
    def __init__(self, root, header):
        x_range = [0]  # currently tiled tiffs are not supported
        y_range = [0]  # currently tiled tiffs are not supported

        self.root = root
        self.tiffs = dict()
        self.dtype = header.dtype
        self.num_channels = header.num_channels

        pattern = "test.000{z}.tiff"  # TODO dont hardcode this

        z_range = [
            detect_value(pattern, file_name, dim="z")[0]
            for file_name in self.list_files()
        ]

        for z in z_range:
            self.tiffs[z] = Image.open(
                os.path.join(self.root, to_file_name(z))
            )  # open is lazy

    def read(self, off, shape):
        if not self.has_only_one_channel():
            # modify the shape to also include the num_channels
            shape = tuple(shape) + tuple([self.num_channels])

        data = np.empty(shape=shape)
        for i, (z, tiff_offset) in enumerate(
            self.calculate_relevant_slices(off, shape)
        ):
            if z in self.tiffs:
                data[:, :, i] = np.array(self.tiffs[z])[tiff_offset]
            else:
                shape_without_z = shape[:2] + shape[3:]
                data[:, :, i] = np.zeros(shape_without_z, self.dtype)

        # convert data into shape with dedicated num_channels (len(data.shape) == 4)
        # this only effects data where previously len(data.shape) was 3 and the num_channel is 1
        # data = data.reshape((-1,) + data.shape[-3:])
        if self.has_only_one_channel():
            data = np.expand_dims(data, 4)  # TODO: make this pretty

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, -1, 0)
        return data

    def write(self, off, data):
        # convert data into shape with dedicated num_channels (len(data.shape) == 4)
        # this only effects data where previously len(data.shape) was 3 and the num_channel is 1
        data = data.reshape((-1,) + data.shape[-3:])

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, 0, -1)

        self.assert_correct_data_format(data)

        for i, (z, tiff_offset) in enumerate(
            self.calculate_relevant_slices(off, data.shape)
        ):
            mode = "RGB" if not self.has_only_one_channel() else None

            # initialize images for z_layers that did not exist before
            if z not in self.tiffs:
                total_shape = [
                    sum(x)
                    for x in zip_longest(data[:, :, i].shape, tiff_offset, fillvalue=0)
                ]
                if self.has_only_one_channel():
                    total_shape = tuple(total_shape)[:-1]  # TODO: make this pretty
                self.tiffs[z] = Image.fromarray(np.zeros(total_shape, self.dtype), mode)

            # write new pixel data into the image
            pixel_data = (
                data[:, :, i] if not self.has_only_one_channel() else data[:, :, i, 0]
            )
            new_region = Image.fromarray(pixel_data, mode)
            self.tiffs[z] = self.tiffs[
                z
            ].copy()  # TODO: investigate why a copy must be used (otherwise the save fails)
            self.tiffs[z].paste(new_region, (tiff_offset[0], tiff_offset[1]))

            # save the new image
            self.tiffs[z].save(os.path.join(self.root, to_file_name(z)))

    def compress(self, dst_path: str, compress_files: bool = False):
        raise NotImplementedError

    def list_files(self):
        file_paths = list(iglob(os.path.join(self.root, "*.tiff")))

        for file_path in file_paths:
            yield os.path.relpath(os.path.normpath(file_path), self.root)

    def close(self):
        for layer_name in self.tiffs:
            self.tiffs[layer_name].close()

    def calculate_relevant_slices(self, offset, shape):
        # TODO: use pattern
        for z in range(offset[2] + 1, offset[2] + shape[2] + 1):
            yield tuple(
                (z, offset[0:2])
            )  # return tuple of important z layers an the x-y offset (without z offset)

    def has_only_one_channel(self):
        return self.num_channels == 1

    def assert_correct_data_format(self, data):
        if not len(data.shape) == 4:
            raise AttributeError(
                "The shape of the provided data does not match the expected shape."
            )
        if not data.shape[3] == self.num_channels:
            raise AttributeError(
                "The shape of the provided data does not match the expected shape. (Expected %d channels)"
                % self.num_channels
            )
        if not np.dtype(data.dtype) == self.dtype:
            raise AttributeError(
                "The type of the provided data does not match the expected type. (Expected np.array of tpye %s)"
                % self.dtype.name
            )

    @staticmethod
    def open(root: str, header=None):
        if header is None:
            header = TiffMagHeader()
        return TiffMag(root, header)

    @staticmethod
    def create(root: str, header):
        raise NotImplementedError  # TODO: what is the difference to open?

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


class TiffMagHeader:
    def __init__(self, pattern="{z}.tiff", dtype=np.dtype("uint8"), num_channels=1):
        self.pattern = pattern
        self.dtype = dtype
        self.num_channels = num_channels

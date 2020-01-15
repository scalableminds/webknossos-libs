import itertools
from typing import Optional, List, Tuple, Set

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


def to_file_name(pattern, x, y, z):
    file_name = pattern
    if x is not None:
        file_name = replace_coordinate(file_name, "x", x)
    if y is not None:
        file_name = replace_coordinate(file_name, "y", y)
    if z is not None:
        file_name = replace_coordinate(file_name, "z", z)
    return file_name


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


def detect_ranges(pattern, file_names):
    extracted_value_pairs = [
        extract_xyz_values(pattern, file_name) for file_name in file_names
    ]
    x_values, y_values, z_values = (
        zip(*extracted_value_pairs) if len(extracted_value_pairs) > 0 else [],
        [],
        [],
    )

    # remove duplicates
    return list(set(x_values)), list(set(y_values)), list(set(z_values))


def extract_xyz_values(pattern, file_name):
    x_value = detect_value(pattern, file_name, dim="x")
    y_value = detect_value(pattern, file_name, dim="y")
    z_value = detect_value(pattern, file_name, dim="z")

    x = None if len(x_value) == 0 else x_value[0]
    y = None if len(y_value) == 0 else y_value[0]
    z = None if len(z_value) == 0 else z_value[0]

    return x, y, z


class TiffMag:
    def __init__(self, root, header):

        self.root = root
        self.tiffs = dict()
        self.header = header

        x_range, y_range, z_range = detect_ranges(
            self.header.pattern, self.list_files()
        )

        available_tiffs = list(itertools.product(x_range, y_range, z_range))

        for xyz in available_tiffs:
            self.tiffs[xyz] = TiffReader.open(
                self.get_file_name_for_layer(xyz)
            )  # open is lazy

    def read(self, off, shape):
        # modify the shape to also include the num_channels
        shape = tuple(shape) + tuple([self.header.num_channels])

        data = np.zeros(shape=shape, dtype=self.header.dtype)
        for (
            xyz,
            _,
            offset_in_output_data,
            offset_in_input_data,
        ) in self.calculate_relevant_slices(off, shape):
            x, y, z = xyz
            z_index_in_data = z - off[2]

            if xyz in self.tiffs:
                # load data and discard the padded data
                loaded_data = np.array(self.tiffs[xyz].read(), self.header.dtype)[
                    offset_in_output_data[0] :, offset_in_output_data[1] :
                ]

                index_slice = [
                    slice(offset_in_input_data[0], offset_in_input_data[0] + loaded_data.shape[0]),
                    slice(offset_in_input_data[1], offset_in_input_data[1] + loaded_data.shape[1]),
                    z_index_in_data,
                ]
                if self.has_only_one_channel():
                    index_slice.append(0)

                # store the loaded data at the right position in 'data'
                data[tuple(index_slice)] = loaded_data

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, -1, 0)
        return data

    def write(self, off, data):  # TODO: maybe update gridShape in properties
        if not len(data.shape) == 3:
            # reformat array to have the channels as the first index (similar to wkw)
            # this is only necessary if the data has a dedicated dimensions for the num_channels
            data = np.moveaxis(data, 0, -1)

        self.assert_correct_data_format(data)

        for (
            xyz,
            shape,
            offset_in_output_data,
            offset_in_input_data,
        ) in self.calculate_relevant_slices(off, data.shape):
            # initialize images for z_layers that did not exist before
            x, y, z = xyz
            z_index_in_input_data = z - off[2]
            if xyz not in self.tiffs:
                # 'output_data_shape' might be bigger than 'shape' because it accounts for padded data
                output_data_shape = [
                    sum(x)
                    for x in zip_longest(shape, offset_in_output_data, fillvalue=0)
                ]

                # initialize an empty image with the right shape
                self.tiffs[xyz] = TiffReader.init_tiff(
                    np.zeros(output_data_shape, self.header.dtype),
                    self.get_file_name_for_layer(xyz),
                )

            # write new pixel data into the image
            pixel_data = data[
                slice(offset_in_input_data[0], offset_in_input_data[0] + shape[0]),
                slice(offset_in_input_data[1], offset_in_input_data[1] + shape[1]),
                z_index_in_input_data,
            ]

            self.tiffs[xyz].merge_with_image(pixel_data, offset_in_output_data)

    def compress(self, dst_path: str, compress_files: bool = False):
        raise NotImplementedError

    def list_files(self):
        file_paths = list(iglob(os.path.join(self.root, "*.tif")))

        for file_path in file_paths:
            yield os.path.relpath(os.path.normpath(file_path), self.root)

    def close(self):
        return

    def calculate_relevant_slices(self, offset, shape):
        tile_size = self.header.tile_size

        max_indices = tuple(i1 + i2 for i1, i2 in zip(offset, shape))

        x_first_index = offset[0] // tile_size[0] if tile_size else None  # floor division
        x_last_index = -(-max_indices[0] // tile_size[0]) if tile_size else None  # ceil division
        x_indices = range(x_first_index, x_last_index) if tile_size else [None]

        y_first_index = offset[1] // tile_size[1] if tile_size else None  # floor division
        y_last_index = -(-max_indices[1] // tile_size[1]) if tile_size else None  # ceil division
        y_indices = range(y_first_index, y_last_index) if tile_size else [None]

        for x in x_indices:
            for y in y_indices:
                for z in range(offset[2], offset[2] + shape[2]):
                    # calculate the offsets and the size for the x and y coordinate
                    tile_shape = shape[0:2] + shape[3:4]
                    offset_in_output_data = tuple(offset[0:2] * np.equal((x, y), (x_first_index, y_first_index)))
                    offset_in_input_data = (0, 0)

                    if tile_size:
                        tile_top_left_corner = np.array((x, y)) * tile_size
                        tile_bottom_right_corner = tile_top_left_corner + tile_size
                        shape_top_left_corner = np.maximum(offset[0:2], tile_top_left_corner)
                        shape_bottom_right = np.minimum(max_indices[0:2], tile_bottom_right_corner)

                        offset_in_input_data = shape_top_left_corner - offset[0:2]
                        tile_shape = tuple(shape_bottom_right - shape_top_left_corner) + tuple(shape[3:4])

                    yield tuple(
                        (
                            (x, y, z),
                            tile_shape,
                            offset_in_output_data,
                            offset_in_input_data,
                        )
                    )

    def has_only_one_channel(self):
        return self.header.num_channels == 1

    def assert_correct_data_format(self, data):
        if self.has_only_one_channel():
            if not len(data.shape) == 3:
                raise AttributeError(
                    "The shape of the provided data does not match the expected shape."
                )
        else:
            if not len(data.shape) == 4:
                raise AttributeError(
                    "The shape of the provided data does not match the expected shape."
                )
            if not data.shape[3] == self.header.num_channels:
                raise AttributeError(
                    "The shape of the provided data does not match the expected shape. (Expected %d channels)"
                    % self.header.num_channels
                )
        if not np.dtype(data.dtype) == self.header.dtype:
            raise AttributeError(
                "The type of the provided data does not match the expected type. (Expected np.array of tpye %s)"
                % self.header.dtype.name
            )

    def get_file_name_for_layer(self, xyz):
        x, y, z = xyz
        return os.path.join(self.root, to_file_name(self.header.pattern, x, y, z))

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
    def __init__(
        self,
        pattern="{z}.tif",
        dtype=np.dtype("uint8"),
        num_channels=1,
        tile_size=(32, 32),
    ):
        self.pattern = pattern
        self.dtype = np.dtype(dtype)
        self.num_channels = num_channels
        self.tile_size = tile_size


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

    def read(self):
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

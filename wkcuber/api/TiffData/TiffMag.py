import itertools
import re
from pathlib import Path
from types import TracebackType
from typing import Optional, List, Tuple, Set, Type, Iterator, cast, Union

from skimage import io
import numpy as np
import os
from glob import iglob
from itertools import zip_longest

from wkcuber.utils import logger


def replace_coordinate(pattern: str, coord_id: str, coord: int) -> str:
    occurrences = re.findall("{" + coord_id + "+}", pattern)
    for occurrence in occurrences:
        number_of_digits = len(occurrence) - 2
        if number_of_digits > 1:
            format_str = "0" + str(number_of_digits) + "d"
        else:
            format_str = "d"
        pattern = pattern.replace(occurrence, format(coord, format_str), 1)
    return pattern


def detect_tile_ranges(
    tiled_dataset_path_parent: Optional[str], tiled_dataset_path_pattern: Optional[str]
) -> Tuple[range, range, range]:
    if tiled_dataset_path_pattern is not None:
        if tiled_dataset_path_parent is not None:
            full_pattern = os.path.join(
                tiled_dataset_path_parent, tiled_dataset_path_pattern
            )
        else:
            full_pattern = tiled_dataset_path_pattern
        pattern_split = os.path.normpath(full_pattern).split(os.path.sep)
        prefix = ""
        if full_pattern.startswith(os.path.sep):
            prefix = "/"

        (
            detected_z_range,
            detected_x_range,
            detected_y_range,
        ) = detect_tile_ranges_from_pattern_recursively(
            pattern_split, prefix, set(), set(), set()
        )

        logger.info(
            f"Auto-detected tile ranges from tif directory structure: z {detected_z_range} x {detected_x_range} y {detected_y_range}"
        )
        return detected_z_range, detected_x_range, detected_y_range

    raise Exception("Couldn't auto-detect tile ranges from wkw or tile path pattern")


def detect_tile_ranges_from_pattern_recursively(
    pattern_elements: List[str],
    prefix: str,
    z_values: Set[int],
    x_values: Set[int],
    y_values: Set[int],
) -> Tuple[range, range, range]:
    (
        current_pattern_element,
        prefix,
        remaining_pattern_elements,
    ) = advance_to_next_relevant_pattern_element(pattern_elements, prefix)
    items = os.listdir(prefix)
    for ls_item in items:
        _, file_extension = os.path.splitext(pattern_elements[-1])
        if (
            os.path.isdir(os.path.join(prefix, ls_item))
            or os.path.splitext(ls_item)[1].lower()[0:4] == file_extension
        ):
            z_values.update(
                detect_value(current_pattern_element, ls_item, "z", ["x", "y"])
            )
            x_values.update(
                detect_value(current_pattern_element, ls_item, "x", ["y", "z"])
            )
            y_values.update(
                detect_value(current_pattern_element, ls_item, "y", ["z", "x"])
            )

    prefix = os.path.join(prefix, current_pattern_element)

    if z_values:
        prefix = replace_coordinate(prefix, "z", min(z_values))
    if x_values:
        prefix = replace_coordinate(prefix, "x", min(x_values))
    if y_values:
        prefix = replace_coordinate(prefix, "y", min(y_values))

    if (
        os.path.exists(prefix)
        and os.path.isdir(prefix)
        and (z_values or x_values or y_values)
    ):
        return detect_tile_ranges_from_pattern_recursively(
            remaining_pattern_elements, prefix, z_values, x_values, y_values
        )
    else:
        return (
            values_to_range(z_values),
            values_to_range(x_values),
            values_to_range(y_values),
        )


def advance_to_next_relevant_pattern_element(
    pattern_elements: List[str], prefix: str
) -> Tuple[str, str, List[str]]:
    current_pattern_element = ""
    i = 0
    for i, pattern_element in enumerate(pattern_elements):
        if "{" in pattern_element or "}" in pattern_element:
            current_pattern_element = pattern_element
            break
        prefix = os.path.join(prefix, pattern_element)
    remaining_pattern_elements = pattern_elements[i + 1 :]
    return current_pattern_element, prefix, remaining_pattern_elements


def values_to_range(values: Set[int]) -> range:
    if len(values) > 0:
        return range(min(values), max(values) + 1)
    return range(0, 0)


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


def to_file_name(
    pattern: str, x: Optional[int], y: Optional[int], z: Optional[int]
) -> str:
    file_name = pattern
    if x is not None:
        file_name = replace_coordinate(file_name, "x", x)
    if y is not None:
        file_name = replace_coordinate(file_name, "y", y)
    if z is not None:
        file_name = replace_coordinate(file_name, "z", z)
    return file_name


class TiffMagHeader:
    def __init__(
        self,
        pattern: str = "{zzzzz}.tif",
        dtype_per_channel: np.dtype = np.dtype("uint8"),
        num_channels: int = 1,
        tile_size: Optional[Tuple[int, int]] = (32, 32),
    ) -> None:
        self.pattern = pattern
        self.dtype_per_channel = np.dtype(dtype_per_channel)
        self.num_channels = num_channels
        self.tile_size = tile_size


class TiffMag:
    def __init__(self, root: str, header: TiffMagHeader) -> None:

        self.root = root
        self.tiffs = dict()
        self.header = header

        detected_z_range, detected_x_range, detected_y_range = detect_tile_ranges(
            self.root, self.header.pattern
        )
        z_range: List[Optional[int]] = (
            [None] if detected_z_range == range(0, 0) else list(detected_z_range)
        )
        y_range: List[Optional[int]] = (
            [None] if detected_y_range == range(0, 0) else list(detected_y_range)
        )
        x_range: List[Optional[int]] = (
            [None] if detected_x_range == range(0, 0) else list(detected_x_range)
        )
        available_tiffs = list(itertools.product(x_range, y_range, z_range))

        for xyz in available_tiffs:
            if xyz != (None, None, None):
                filename = self.get_file_name_for_layer(xyz)
                if Path(filename).is_file():
                    self.tiffs[xyz] = TiffReader.open(
                        self.get_file_name_for_layer(xyz)
                    )  # open is lazy

    def read(self, off: Tuple[int, int, int], shape: Tuple[int, int, int]) -> np.array:
        # modify the shape to also include the num_channels
        shape_with_num_channels = shape + (self.header.num_channels,)
        data = np.zeros(
            shape=shape_with_num_channels, dtype=self.header.dtype_per_channel
        )
        for (
            xyz,
            _,
            offset_in_output_data,
            offset_in_input_data,
        ) in self.calculate_relevant_slices(off, shape_with_num_channels):
            _, _, z = xyz
            z_index_in_data = z - off[2]

            if xyz in self.tiffs:
                # load data and discard the padded data
                loaded_data = np.array(
                    self.tiffs[xyz].read(), self.header.dtype_per_channel
                )[
                    offset_in_output_data[0] : offset_in_output_data[0]
                    + shape_with_num_channels[0],
                    offset_in_output_data[1] : offset_in_output_data[1]
                    + shape_with_num_channels[1],
                ]

                index_slice = [
                    slice(
                        offset_in_input_data[0],
                        offset_in_input_data[0] + loaded_data.shape[0],
                    ),
                    slice(
                        offset_in_input_data[1],
                        offset_in_input_data[1] + loaded_data.shape[1],
                    ),
                    z_index_in_data,
                ]
                if self.has_only_one_channel():
                    index_slice.append(0)

                # store the loaded data at the right position in 'data'
                data[tuple(index_slice)] = loaded_data

        # reformat array to have the channels as the first index (similar to wkw)
        data = np.moveaxis(data, -1, 0)
        return data

    def write(self, off: Tuple[int, int, int], data: np.ndarray) -> None:
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
                    np.zeros(output_data_shape, self.header.dtype_per_channel),
                    self.get_file_name_for_layer(xyz),
                )

            # write new pixel data into the image
            pixel_data = data[
                slice(offset_in_input_data[0], offset_in_input_data[0] + shape[0]),
                slice(offset_in_input_data[1], offset_in_input_data[1] + shape[1]),
                z_index_in_input_data,
            ]

            self.tiffs[xyz].merge_with_image(pixel_data, offset_in_output_data)

    def list_files(self) -> Iterator[str]:
        _, file_extension = os.path.splitext(self.header.pattern)
        return iglob(
            self.root + "/" + re.sub(r"{.*?}", "*", self.header.pattern), recursive=True
        )

    def close(self) -> None:
        return

    def calculate_relevant_slices(
        self,
        offset: Tuple[int, int, int],
        shape: Union[Tuple[int, int, int, int], Tuple[int, int, int]],
    ) -> Iterator[
        Tuple[
            Tuple[Optional[int], Optional[int], int],
            Tuple[int, ...],
            Tuple[int, int],
            Tuple[int, int],
        ]
    ]:
        """
        The purpose of this method is to find out which tiles need to be touched.
        Each tile is specified by its (x, y, z)-dimensions.
        For each tile, this method also returns what offsets inside of each individual tile need to be used.
        Additionally, this method returns for each tile where the data of the tile fits into the bigger picture.
        :param offset: the offset in the dataset compared to the coordinate (0, 0, 0)
        :param shape: the shape of the data that is about to be written or read (depending on where this method is used)
        :return: tiles that need to be considered (+ their shape, the offset in the tiles, and the offset in the original data)
        """

        tile_size = (
            self.header.tile_size
        )  # tile_size is None if the dataset is a simple TiffDataset

        max_indices = tuple(i1 + i2 for i1, i2 in zip(offset, shape))

        if tile_size is None:
            x_first_index = None
            x_indices: List[Union[int, None]] = [None]
            y_first_index = None
            y_indices: List[Union[int, None]] = [None]
        else:
            x_first_index = offset[0] // tile_size[0]  # floor division
            x_last_index = np.math.ceil(max_indices[0] / tile_size[0])
            x_indices = list(range(x_first_index, x_last_index))

            y_first_index = offset[1] // tile_size[1]  # floor division
            y_last_index = np.math.ceil(max_indices[1] / tile_size[1])
            y_indices = list(range(y_first_index, y_last_index))

        for x in x_indices:
            for y in y_indices:
                for z in range(offset[2], offset[2] + shape[2]):
                    # calculate the offsets and the size for the x and y coordinate
                    tile_shape = shape[0:2] + shape[3:4]
                    offset_in_output_data = tuple(
                        offset[0:2] * np.equal((x, y), (x_first_index, y_first_index))
                    )
                    offset_in_input_data = (0, 0)

                    if tile_size:
                        tile_top_left_corner = np.array((x, y)) * tile_size
                        tile_bottom_right_corner = tile_top_left_corner + tile_size
                        shape_top_left_corner = np.maximum(
                            offset[0:2], tile_top_left_corner
                        )
                        shape_bottom_right = np.minimum(
                            max_indices[0:2], tile_bottom_right_corner
                        )

                        offset_in_input_data = shape_top_left_corner - offset[0:2]
                        offset_in_output_data = tuple(
                            (np.array(offset[0:2]) - shape_top_left_corner)
                            * np.equal((x, y), (x_first_index, y_first_index))
                        )
                        tile_shape = (
                            tuple(shape_bottom_right - shape_top_left_corner)
                            + shape[3:4]
                        )

                    yield (
                        (x, y, z),
                        tile_shape,
                        cast(Tuple[int, int], offset_in_output_data),
                        offset_in_input_data,
                    )

    def has_only_one_channel(self) -> bool:
        return self.header.num_channels == 1

    def assert_correct_data_format(self, data: np.ndarray) -> None:
        if self.has_only_one_channel():
            if not len(data.shape) == 3:
                raise AttributeError(
                    "The shape of the provided data does not match the expected shape. Expected three-dimensional data shape, since target data is single-channel."
                )
        else:
            if not len(data.shape) == 4:
                raise AttributeError(
                    "The shape of the provided data does not match the expected shape."
                )
            if not data.shape[3] == self.header.num_channels:
                raise AttributeError(
                    f"The shape of the provided data does not match the expected shape. (Expected {self.header.num_channels} channels)"
                )
        if not np.dtype(data.dtype) == self.header.dtype_per_channel:
            raise AttributeError(
                f"The type of the provided data does not match the expected type. (Expected np.array of type {self.header.dtype_per_channel.name})"
            )

    def get_file_name_for_layer(
        self, xyz: Tuple[Optional[int], Optional[int], Optional[int]]
    ) -> str:
        x, y, z = xyz
        return os.path.join(self.root, to_file_name(self.header.pattern, x, y, z))

    @staticmethod
    def open(root: str, header: TiffMagHeader = None) -> "TiffMag":
        if header is None:
            header = TiffMagHeader()
        return TiffMag(root, header)

    def __enter__(self) -> "TiffMag":
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self.close()


def transpose_for_skimage(data: np.ndarray) -> np.ndarray:
    if len(data.shape) == 2:
        return data.transpose()
    elif len(data.shape) == 3:
        return data.transpose((1, 0, 2))
    else:
        raise ValueError("Cannot handle shape for data.")


class TiffReader:
    def __init__(self, file_name: str):
        self.file_name = file_name

    @classmethod
    def init_tiff(cls, pixels: np.ndarray, file_name: str) -> "TiffReader":
        tr = TiffReader(file_name)
        tr.write(pixels)
        return tr

    @classmethod
    def open(cls, file_name: str) -> "TiffReader":
        return cls(file_name)

    def read(self) -> np.array:
        data = io.imread(self.file_name)
        return transpose_for_skimage(data)

    def write(self, pixels: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self.file_name), exist_ok=True)
        io.imsave(self.file_name, transpose_for_skimage(pixels), check_contrast=False)

    def merge_with_image(
        self, foreground_pixels: np.ndarray, offset: Tuple[int, int]
    ) -> None:
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

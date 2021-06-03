import logging
import os
import shutil
from argparse import Namespace
from os.path import join
from pathlib import Path
from typing import (
    Tuple,
    Union,
    cast,
    TYPE_CHECKING,
    Generator,
)
from uuid import uuid4

from wkw import wkw
import numpy as np

from wkcuber.compress_utils import compress_file_job
from wkcuber.utils import (
    convert_mag1_size,
    convert_mag1_offset,
    get_executor_for_args,
    wait_and_ensure_success,
)

if TYPE_CHECKING:
    from wkcuber.api.layer import (
        Layer,
    )
from wkcuber.api.view import View
from wkcuber.mag import Mag


def _find_mag_path_on_disk(dataset_path: Path, layer_name: str, mag_name: str) -> Path:
    mag = Mag(mag_name)
    short_mag_file_path = dataset_path / layer_name / mag.to_layer_name()
    long_mag_file_path = dataset_path / layer_name / mag.to_long_layer_name()
    if os.path.exists(short_mag_file_path):
        return short_mag_file_path
    else:
        return long_mag_file_path


class WKMagDataset:
    """
    A `WKMagDataset` contains all information about the data of a single magnification of a `wkcuber.api.layer.Layer`.
    """

    def __init__(
        self,
        layer: "Layer",
        name: str,
        block_len: int,
        file_len: int,
        block_type: int,
        create: bool = False,
    ) -> None:
        self.layer = layer
        self.name = name
        self.block_len = block_len
        self.file_len = file_len
        self.block_type = block_type
        self.header: wkw.Header = wkw.Header(
            voxel_type=self.layer.dtype_per_channel,
            num_channels=self.layer.num_channels,
            version=1,
            block_len=self.block_len,
            file_len=self.file_len,
            block_type=self.block_type,
        )

        self.view = self.get_view(offset=(0, 0, 0), is_bounded=False)

        if create:
            wkw.Dataset.create(
                join(layer.dataset.path, layer.name, self.name), self.header
            )

    def open(self) -> None:
        """
        Opens the actual handles to the data on disk.
        A `MagDataset` has to be opened before it can be read or written to. However, the user does not
        have to open it explicitly because the API automatically opens it when it is needed.
        The user can choose to open it explicitly to avoid that handles are opened and closed automatically
        each time data is read or written.
        """
        self.view.open()

    def close(self) -> None:
        """
        Complementary to `open`, this closes the handles to the data.

        See `open` for more information.
        """
        self.view.close()

    def read(
        self,
        offset: Tuple[int, int, int] = (0, 0, 0),
        size: Tuple[int, int, int] = None,
    ) -> np.array:
        """
        The user can specify the `offset` and the `size` of the requested data.
        The `offset` refers to the absolute position, regardless of the offset in the properties.
        If no `size` is specified, the offset from the properties + the size of the properties is used.
        If the specified bounding box exceeds the data on disk, the rest is padded with `0`.

        Retruns the specified data as a `np.array`.
        """
        return self.view.read(offset, size)

    def write(
        self,
        data: np.ndarray,
        offset: Tuple[int, int, int] = (0, 0, 0),
        allow_compressed_write: bool = False,
    ) -> None:
        """
        Writes the `data` at the specified `offset` to disk.
        The `offset` refers to the absolute position, regardless of the offset in the properties.
        If the data exceeds the previous bounding box, the properties are updated.
        If the data on disk is compressed, the passed `data` either has to be aligned with the files on disk
        or `allow_compressed_write` has to be `True`. If `allow_compressed_write` is `True`, `data` is padded by
        first reading the necessary padding from disk.
        """
        self._assert_valid_num_channels(data.shape)
        self.view.write(data, offset, allow_compressed_write)
        layer_properties = self.layer.dataset.properties.data_layers[self.layer.name]
        current_offset_in_mag1 = layer_properties.get_bounding_box_offset()
        current_size_in_mag1 = layer_properties.get_bounding_box_size()

        mag = Mag(self.name)
        mag_np = mag.as_np()

        offset_in_mag1 = tuple(np.array(offset) * mag_np)

        new_offset_in_mag1 = (
            offset_in_mag1
            if current_offset_in_mag1 == (-1, -1, -1)
            else tuple(min(x) for x in zip(current_offset_in_mag1, offset_in_mag1))
        )

        old_end_offset_in_mag1 = np.array(current_offset_in_mag1) + np.array(
            current_size_in_mag1
        )
        new_end_offset_in_mag1 = (np.array(offset) + np.array(data.shape[-3:])) * mag_np
        max_end_offset_in_mag1 = np.array(
            [old_end_offset_in_mag1, new_end_offset_in_mag1]
        ).max(axis=0)
        total_size_in_mag1 = max_end_offset_in_mag1 - np.array(new_offset_in_mag1)

        self.view.size = cast(
            Tuple[int, int, int],
            tuple(convert_mag1_offset(max_end_offset_in_mag1, mag)),
        )  # The base view of a MagDataset always starts at (0, 0, 0)

        self.layer.dataset.properties._set_bounding_box_of_layer(
            self.layer.name,
            cast(Tuple[int, int, int], tuple(new_offset_in_mag1)),
            cast(Tuple[int, int, int], tuple(total_size_in_mag1)),
        )

    def get_view(
        self,
        size: Tuple[int, int, int] = None,
        offset: Tuple[int, int, int] = None,
        is_bounded: bool = True,
        read_only: bool = False,
    ) -> View:
        """
        Returns a view that is limited to the specified bounding box.
        The default value for `offset` is the offset that is specified in the properties.
        The default value for `size` is calculated so that the bounding box ends where the bounding box from the
        properties ends.
        Therefore, if both (`offset` and `size`) are not specified, then the bounding box of the view is equal to the
        bounding box specified in the properties.
        If `is_bounded` is `True`, reading or writing outside of this bounding box is not allowed.
        """
        mag1_size_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_size()

        mag1_offset_in_properties = self.layer.dataset.properties.data_layers[
            self.layer.name
        ].get_bounding_box_offset()

        if mag1_offset_in_properties == (-1, -1, -1):
            mag1_offset_in_properties = (0, 0, 0)

        view_offset = (
            offset
            if offset is not None
            else cast(
                Tuple[int, int, int],
                tuple(convert_mag1_offset(mag1_offset_in_properties, Mag(self.name))),
            )
        )

        properties_offset_in_current_mag = convert_mag1_offset(
            mag1_offset_in_properties, Mag(self.name)
        )

        if size is None:
            size = convert_mag1_size(mag1_size_in_properties, Mag(self.name)) - (
                np.array(view_offset) - properties_offset_in_current_mag
            )

        # assert that the parameters size and offset are valid
        if is_bounded:
            for off_prop, off in zip(properties_offset_in_current_mag, view_offset):
                if off < off_prop:
                    raise AssertionError(
                        f"The passed parameter 'offset' {view_offset} is outside the bounding box from the properties.json. "
                        f"Use is_bounded=False if you intend to write outside out the existing bounding box."
                    )
            for s1, s2, off1, off2 in zip(
                convert_mag1_size(mag1_size_in_properties, Mag(self.name)),
                size,
                properties_offset_in_current_mag,
                view_offset,
            ):
                if s2 + off2 > s1 + off1:
                    raise AssertionError(
                        f"The combination of the passed parameter 'size' {size} and 'offset' {view_offset} are not compatible with the "
                        f"size ({mag1_size_in_properties}) from the properties.json.  "
                        f"Use is_bounded=False if you intend to write outside out the existing bounding box."
                    )

        mag_file_path = _find_mag_path_on_disk(
            self.layer.dataset.path, self.layer.name, self.name
        )
        return View(
            mag_file_path,
            self.header,
            cast(Tuple[int, int, int], tuple(size)),
            view_offset,
            is_bounded,
            read_only,
        )

    def _assert_valid_num_channels(self, write_data_shape: Tuple[int, ...]) -> None:
        num_channels = self.layer.num_channels
        if len(write_data_shape) == 3:
            assert (
                num_channels == 1
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data (1)"
        else:
            assert (
                num_channels == write_data_shape[0]
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data ({write_data_shape[0]})"

    def get_bounding_boxes_on_disk(
        self,
    ) -> Generator[Tuple[Tuple[int, int, int], Tuple[int, int, int]], None, None]:
        """
        Returns a bounding box for each file on disk.
        A bounding box is represented as a tuple of the offset and the size.

        This differs from the bounding box in the properties in two ways:
        - the bounding box in the properties is always specified in mag 1
        - the bounding box in the properties is an "overall" bounding box, which abstracts from the files on disk
        """
        cube_size = self._get_file_dimensions()
        was_opened = self.view._is_opened

        if not was_opened:
            self.open()  # opening the view is necessary to set the dataset

        assert self.view.dataset is not None
        for filename in self.view.dataset.list_files():
            file_path = Path(os.path.splitext(filename)[0]).relative_to(self.view.path)
            cube_index = _extract_file_index(file_path)
            cube_offset = [idx * size for idx, size in zip(cube_index, cube_size)]

            yield (cube_offset[0], cube_offset[1], cube_offset[2]), cube_size

        if not was_opened:
            self.close()

    def compress(
        self, target_path: Union[str, Path] = None, args: Namespace = None
    ) -> None:
        """
        Compresses the files on disk. This has consequences for writing data (see `write`).

        The data gets compressed inplace, if target_path is None.
        Otherwise it is written to target_path/layer_name/mag.
        """

        if target_path is not None:
            target_path = Path(target_path)

        uncompressed_full_path = (
            Path(self.layer.dataset.path) / self.layer.name / str(Mag(self.name))
        )
        compressed_path = (
            target_path
            if target_path is not None
            else Path("{}.compress-{}".format(self.layer.dataset.path, uuid4()))
        )
        compressed_full_path = compressed_path / self.layer.name / str(Mag(self.name))

        if compressed_full_path.exists():
            logging.error(
                "Target path '{}' already exists".format(compressed_full_path)
            )
            exit(1)

        logging.info(
            "Compressing mag {0} in '{1}'".format(
                self.name, str(uncompressed_full_path)
            )
        )

        was_opened = self.view._is_opened
        if not was_opened:
            self.open()  # opening the view is necessary to set the dataset
        assert self.view.dataset is not None

        # create empty wkw.Dataset
        self.view.dataset.compress(str(compressed_full_path))

        # compress all files to and move them to 'compressed_path'
        with get_executor_for_args(args) as executor:
            job_args = []
            for file in self.view.dataset.list_files():
                rel_file = Path(file).relative_to(self.layer.dataset.path)
                job_args.append((Path(file), compressed_path / rel_file))

            wait_and_ensure_success(
                executor.map_to_futures(compress_file_job, job_args)
            )

        logging.info("Mag {0} successfully compressed".format(self.name))

        if not was_opened:
            self.close()

        if target_path is None:
            shutil.rmtree(uncompressed_full_path)
            shutil.move(str(compressed_full_path), uncompressed_full_path)
            shutil.rmtree(compressed_path)

            # update the handle to the new dataset
            self.view = self.get_view(offset=(0, 0, 0), is_bounded=False)

    def _get_file_dimensions(self) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], (self.file_len * self.block_len,) * 3)


def _extract_file_index(file_path: Path) -> Tuple[int, int, int]:
    zyx_index = [int(el[1:]) for el in file_path.parts]
    return zyx_index[2], zyx_index[1], zyx_index[0]

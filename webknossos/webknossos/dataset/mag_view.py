import logging
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple, Union, cast
from uuid import uuid4

import numpy as np
from wkw import wkw

from webknossos.geometry import BoundingBox, Mag, Vec3Int, Vec3IntLike
from webknossos.utils import get_executor_for_args, wait_and_ensure_success

from .compress_utils import compress_file_job
from .properties import MagViewProperties

if TYPE_CHECKING:
    from .layer import (
        Layer,
    )

from .view import View


def _find_mag_path_on_disk(dataset_path: Path, layer_name: str, mag_name: str) -> Path:
    mag = Mag(mag_name)
    short_mag_file_path = dataset_path / layer_name / mag.to_layer_name()
    long_mag_file_path = dataset_path / layer_name / mag.to_long_layer_name()
    if os.path.exists(short_mag_file_path):
        return short_mag_file_path
    else:
        return long_mag_file_path


def _convert_mag1_offset(
    mag1_offset: Union[List, np.ndarray], target_mag: Mag
) -> np.ndarray:
    return np.array(mag1_offset) // target_mag.to_np()  # floor div


class MagView(View):
    """
    A `MagView` contains all information about the data of a single magnification of a `webknossos.dataset.layer.Layer`.
    `MagView` inherits from `webknossos.dataset.view.View`.
    Therefore, the main difference between them is that a `MagView` handles the whole magnification,
    whereas the `View` only handles a sub-region.

    A `MagView` can read/write outside the specified bounding box (unlike a normal `View`).
    If necessary, the properties are automatically updated (e.g. if the bounding box changed).
    This is possible because a `MagView` does have a reference to the `webknossos.dataset.layer.Layer`.

    The `global_offset` of a `MagView` is always `(0, 0, 0)` and its `size` is chosen so that the bounding box from the properties is fully inside this View.
    """

    def __init__(
        self,
        layer: "Layer",
        mag: Mag,
        block_len: int,
        file_len: int,
        block_type: int,
        create: bool = False,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.add_mag()` to create a `MagView`.
        """
        header = wkw.Header(
            voxel_type=layer.dtype_per_channel,
            num_channels=layer.num_channels,
            version=1,
            block_len=block_len,
            file_len=file_len,
            block_type=block_type,
        )

        self._layer = layer
        self._mag = mag

        super().__init__(
            _find_mag_path_on_disk(
                self.layer.dataset.path, self.layer.name, mag.to_layer_name()
            ),
            header,
            cast(
                Tuple[int, int, int],
                tuple(self._mag_view_bounding_box_at_creation.bottomright),
            ),
            (0, 0, 0),
            False,
            False,
            None,
        )

        if create:
            wkw.Dataset.create(str(self.path), self.header)

    @property
    def layer(self) -> "Layer":
        return self._layer

    @property
    def path(self) -> Path:
        return self._path

    @property
    def _properties(self) -> MagViewProperties:
        return next(
            mag_property
            for mag_property in self.layer._properties.wkw_resolutions
            if mag_property.resolution == self.mag
        )

    @property
    def name(self) -> str:
        return self._mag.to_layer_name()

    @property
    def mag(self) -> Mag:
        return self._mag

    def write(self, data: np.ndarray, offset: Vec3IntLike = Vec3Int(0, 0, 0)) -> None:
        """
        Writes the `data` at the specified `offset` to disk (like `webknossos.dataset.view.View.write()`).

        The `offset` refers to the absolute position, regardless of the offset in the properties (because the global_offset is set to (0, 0, 0)).
        If the data exceeds the original bounding box, the properties are updated.

        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions.
        """
        offset = Vec3Int(offset)

        self._assert_valid_num_channels(data.shape)
        super().write(data, offset)
        current_offset_in_mag1 = self.layer.bounding_box.topleft.to_np()
        current_size_in_mag1 = self.layer.bounding_box.size.to_np()

        mag_np = self.mag.to_np()

        offset_in_mag1 = offset.to_np() * mag_np

        # The (-1, -1, -1) is for backwards compatibility because we used '(-1, -1, -1)' to indicate that there is no data written yet.
        new_offset_in_mag1 = (
            offset_in_mag1
            if tuple(current_offset_in_mag1) == (-1, -1, -1)
            or self.layer.bounding_box == BoundingBox((0, 0, 0), (0, 0, 0))
            else np.minimum(current_offset_in_mag1, offset_in_mag1)
        )

        old_end_offset_in_mag1 = current_offset_in_mag1 + current_size_in_mag1
        new_end_offset_in_mag1 = (np.array(offset) + np.array(data.shape[-3:])) * mag_np
        max_end_offset_in_mag1 = np.array(
            [old_end_offset_in_mag1, new_end_offset_in_mag1]
        ).max(axis=0)
        total_size_in_mag1 = max_end_offset_in_mag1 - np.array(new_offset_in_mag1)

        # The base view of a MagDataset always starts at (0, 0, 0)
        self._size = Vec3Int(_convert_mag1_offset(max_end_offset_in_mag1, self.mag))

        self.layer.bounding_box = BoundingBox(
            new_offset_in_mag1,
            total_size_in_mag1,
        )

    def get_view(
        self,
        offset: Optional[Vec3IntLike] = None,
        size: Optional[Vec3IntLike] = None,
        read_only: Optional[bool] = None,
    ) -> View:
        """
        Returns a view that is limited to the specified bounding box.

        The `offset` refers to the absolute position, regardless of the offset in the properties (because the global_offset is set to (0, 0, 0)).
        The default value for `offset` is the offset that is specified in the properties.
        The default value for `size` is calculated so that the bounding box ends where the bounding box from the
        properties ends.
        Therefore, if both (`offset` and `size`) are not specified, then the bounding box of the view is equal to the
        bounding box specified in the properties.

        The `offset` and `size` may only exceed the bounding box from the properties, if `read_only` is set to `True`.

        If `read_only` is `True`, write operations are not allowed for the returned sub-view.

        Example:
        ```python
        # ...
        # Let 'mag1' be a `MagView` with offset (0, 0, 0) and size (100, 200, 300)

        # Properties are used to determine the default parameter
        view_with_bb_from_properties = mag1.get_view()

        # Sub-view where the specified bounding box is completely in the bounding box of the MagView
        sub_view1 = mag1.get_view(offset=(50, 60, 70), size=(10, 120, 230))

        # Fails because the specified view is not completely in the bounding box from the properties.
        sub_view2 = mag1.get_view(offset=(50, 60, 70), size=(999, 120, 230), read_only=True)

        # Sub-view where the specified bounding box is NOT completely in the bounding box of the MagView.
        # This still works because `read_only=True`.
        sub_view2 = mag1.get_view(offset=(50, 60, 70), size=(999, 120, 230), read_only=True)
        ```
        """

        bb = self.layer.bounding_box

        # The (-1, -1, -1) is for backwards compatibility because we used '(-1, -1, -1)' to indicate that there is no data written yet.
        if bb.topleft == Vec3Int(-1, -1, -1):
            bb = bb.with_topleft((0, 0, 0))

        bb = bb.align_with_mag(self.mag, ceil=True).in_mag(self.mag)

        offset = Vec3Int(offset) if offset is not None else bb.topleft
        size = Vec3Int(size) if size is not None else bb.bottomright - offset

        assert bb.contains_bbox(BoundingBox(offset, size)) or read_only
        return super().get_view(
            offset,
            size,
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
        for filename in self._wkw_dataset.list_files():
            file_path = Path(os.path.splitext(filename)[0]).relative_to(self._path)
            cube_index = _extract_file_index(file_path)
            cube_offset = [idx * size for idx, size in zip(cube_index, cube_size)]

            yield (cube_offset[0], cube_offset[1], cube_offset[2]), cube_size

    def compress(
        self,
        target_path: Optional[Union[str, Path]] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Compresses the files on disk. This has consequences for writing data (see `write`).

        The data gets compressed inplace, if target_path is None.
        Otherwise it is written to target_path/layer_name/mag.
        """

        if target_path is not None:
            target_path = Path(target_path)

        uncompressed_full_path = (
            Path(self.layer.dataset.path) / self.layer.name / self.name
        )
        compressed_path = (
            target_path
            if target_path is not None
            else Path("{}.compress-{}".format(self.layer.dataset.path, uuid4()))
        )
        compressed_full_path = compressed_path / self.layer.name / self.name

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
        # create empty wkw.Dataset
        self._wkw_dataset.compress(str(compressed_full_path))

        # compress all files to and move them to 'compressed_path'
        with get_executor_for_args(args) as executor:
            job_args = []
            for file in self._wkw_dataset.list_files():
                rel_file = Path(file).relative_to(self.layer.dataset.path)
                job_args.append((Path(file), compressed_path / rel_file))

            wait_and_ensure_success(
                executor.map_to_futures(compress_file_job, job_args)
            )

        logging.info("Mag {0} successfully compressed".format(self.name))

        if target_path is None:
            shutil.rmtree(uncompressed_full_path)
            shutil.move(str(compressed_full_path), uncompressed_full_path)
            shutil.rmtree(compressed_path)

            # update the handle to the new dataset
            MagView.__init__(
                self,
                self.layer,
                self.mag,
                self.header.block_len,
                self.header.file_len,
                wkw.Header.BLOCK_TYPE_LZ4HC,
            )

    def _get_file_dimensions(self) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], (self._properties.cube_length,) * 3)

    @property
    def _mag_view_bounding_box_at_creation(self) -> BoundingBox:
        return self.layer.bounding_box.align_with_mag(Mag(self.name), ceil=True).in_mag(
            Mag(self.name)
        )

    def __repr__(self) -> str:
        return repr(
            "MagView(name=%s, global_offset=%s, size=%s)"
            % (self.name, self.global_offset, self.size)
        )


def _extract_file_index(file_path: Path) -> Tuple[int, int, int]:
    zyx_index = [int(el[1:]) for el in file_path.parts]
    return zyx_index[2], zyx_index[1], zyx_index[0]

import logging
import os
import shutil
import warnings
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional, Union
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


class MagView(View):
    """
    A `MagView` contains all information about the data of a single magnification of a `Layer`.
    `MagView` inherits from `View`. The main difference is that the `MagView `has a reference to its `Layer`

    Therefore, a `MagView` can write outside the specified bounding box (unlike a normal `View`), resizing the layer's bounding box.
    If necessary, the properties are automatically updated (e.g. if the bounding box changed).
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
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.add_mag()`.
        """
        header = wkw.Header(
            voxel_type=layer.dtype_per_channel,
            num_channels=layer.num_channels,
            version=1,
            block_len=block_len,
            file_len=file_len,
            block_type=block_type,
        )

        super().__init__(
            _find_mag_path_on_disk(layer.dataset.path, layer.name, mag.to_layer_name()),
            header,
            bounding_box=None,
            mag=mag,
        )
        self._layer = layer

        if create:
            wkw.Dataset.create(str(self.path), self.header)

    # Overwrites of View methods:
    @property
    def bounding_box(self) -> BoundingBox:
        # Overwrites View's method since no extra bbox is stored for a MagView,
        # but the Layer's bbox is used:
        return self.layer.bounding_box.align_with_mag(self._mag, ceil=True)

    @property
    def global_offset(self) -> Vec3Int:
        """⚠️ Deprecated, use `Vec3Int.zeros()` instead."""
        warnings.warn(
            "[DEPRECATION] mag_view.global_offset is deprecated. "
            + "Since this is a MagView, please use "
            + "Vec3Int.zeros() instead."
        )
        return Vec3Int.zeros()

    @property
    def size(self) -> Vec3Int:
        """⚠️ Deprecated, use `mag_view.bounding_box.in_mag(mag_view.mag).bottomright` instead."""
        warnings.warn(
            "[DEPRECATION] mag_view.size is deprecated. "
            + "Since this is a MagView, please use "
            + "mag_view.bounding_box.in_mag(mag_view.mag).bottomright instead."
        )
        return self.bounding_box.in_mag(self._mag).bottomright

    # Own methods:

    @property
    def layer(self) -> "Layer":
        return self._layer

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._mag.to_layer_name()

    def write(
        self,
        data: np.ndarray,
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative, in current mag
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
    ) -> None:
        if offset is not None:
            if self._mag == Mag(1):
                alternative = "Since this is a MagView in Mag(1), please use mag_view.write(absolute_offset=my_vec)"
            else:
                alternative = (
                    "Since this is a MagView, please use the coordinates in Mag(1) instead, e.g. "
                    + "mag_view.write(absolute_offset=my_vec * mag_view.mag.to_vec3_int())"
                )

            warnings.warn(
                "[DEPRECATION] Using mag_view.write(offset=my_vec) is deprecated. "
                + "Please use relative_offset or absolute_offset instead. "
                + alternative
            )

        if all(i is None for i in [offset, absolute_offset, relative_offset]):
            relative_offset = Vec3Int.zeros()

        mag1_bbox = self._get_mag1_bbox(
            abs_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            current_mag_size=Vec3Int(data.shape[-3:]),
        )
        self.layer.bounding_box = self.layer.bounding_box.extended_by(mag1_bbox)

        super().write(data, absolute_offset=mag1_bbox.topleft)

    def read(
        self,
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative, in current mag
        size: Optional[
            Vec3IntLike
        ] = None,  # usually in mag1, in current mag if offset is given
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        relative_bounding_box: Optional[BoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[BoundingBox] = None,  # in mag1
    ) -> np.ndarray:
        # THIS METHOD CAN BE REMOVED WHEN THE DEPRECATED OFFSET IS REMOVED

        if (
            relative_offset is not None
            or absolute_offset is not None
            or absolute_bounding_box is not None
            or relative_bounding_box is not None
        ) or (
            offset is None
            and size is None
            and relative_offset is None
            and absolute_offset is None
            and absolute_bounding_box is None
            and relative_bounding_box is None
        ):
            return super().read(
                offset,
                size,
                relative_offset=relative_offset,
                absolute_offset=absolute_offset,
                absolute_bounding_box=absolute_bounding_box,
                relative_bounding_box=relative_bounding_box,
            )
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="webknossos"
                )
                view = self.get_view(offset=Vec3Int.zeros(), read_only=True)
            return view.read(offset, size)

    def get_view(
        self,
        offset: Optional[Vec3IntLike] = None,
        size: Optional[Vec3IntLike] = None,
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        read_only: Optional[bool] = None,
    ) -> View:
        # THIS METHOD CAN BE REMOVED WHEN THE DEPRECATED OFFSET IS REMOVED

        # This has other defaults than the View implementation
        # (all deprecations are handled in the subsclass)
        bb = self.bounding_box.in_mag(self._mag)
        if offset is not None and size is None:
            offset = Vec3Int(offset)
            size = bb.bottomright - offset

        return super().get_view(
            None if offset is None else Vec3Int(offset) - bb.topleft,
            size,
            relative_offset=relative_offset,
            absolute_offset=absolute_offset,
            read_only=read_only,
        )

    def get_bounding_boxes_on_disk(
        self,
    ) -> Generator[BoundingBox, None, None]:
        """
        Returns a Mag(1) bounding box for each file on disk.

        This differs from the bounding box in the properties, which is an "overall" bounding box,
        abstracting from the files on disk.
        """
        cube_size = self._get_file_dimensions()
        for filename in self._wkw_dataset.list_files():
            file_path = Path(os.path.splitext(filename)[0]).relative_to(self._path)
            cube_index = _extract_file_index(file_path)
            cube_offset = cube_index * cube_size

            yield BoundingBox(cube_offset, cube_size).from_mag_to_mag1(self._mag)

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
                executor.map_to_futures(compress_file_job, job_args), "Compressing"
            )

        if target_path is None:
            shutil.rmtree(uncompressed_full_path)
            shutil.move(str(compressed_full_path), uncompressed_full_path)
            shutil.rmtree(compressed_path)

            # update the handle to the new dataset
            MagView.__init__(
                self,
                self.layer,
                self._mag,
                self.header.block_len,
                self.header.file_len,
                wkw.Header.BLOCK_TYPE_LZ4HC,
            )

    @property
    def _properties(self) -> MagViewProperties:
        return next(
            mag_property
            for mag_property in self.layer._properties.wkw_resolutions
            if mag_property.resolution == self._mag
        )

    def _get_file_dimensions(self) -> Vec3Int:
        return Vec3Int.full(self._properties.cube_length)

    def __repr__(self) -> str:
        return repr(f"MagView(name={self.name}, bounding_box={self.bounding_box})")


def _extract_file_index(file_path: Path) -> Vec3Int:
    zyx_index = [int(el[1:]) for el in file_path.parts]
    return Vec3Int(zyx_index[2], zyx_index[1], zyx_index[0])

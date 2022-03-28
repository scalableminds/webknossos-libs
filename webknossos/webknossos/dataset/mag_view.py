import logging
import warnings
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from ..geometry import BoundingBox, Mag, Vec3Int, Vec3IntLike
from ..utils import (
    get_executor_for_args,
    is_fs_path,
    make_upath,
    rmtree,
    wait_and_ensure_success,
)
from ._array import ArrayInfo, BaseArray
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
    if short_mag_file_path.exists():
        return short_mag_file_path
    else:
        return long_mag_file_path


def _compress_cube_job(args: Tuple[View, View]) -> None:
    source_view, target_view = args
    target_view.write(source_view.read())


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
        chunk_size: Vec3Int,
        chunks_per_shard: Vec3Int,
        compression_mode: bool,
        create: bool = False,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.add_mag()`.
        """
        array_info = ArrayInfo(
            data_format=layer._properties.data_format,
            voxel_type=layer.dtype_per_channel,
            num_channels=layer.num_channels,
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            compression_mode=compression_mode,
        )
        if create:
            self_path = layer.dataset.path / layer.name / mag.to_layer_name()
            BaseArray.get_class(array_info.data_format).create(self_path, array_info)

        super().__init__(
            _find_mag_path_on_disk(layer.dataset.path, layer.name, mag.to_layer_name()),
            array_info,
            bounding_box=None,
            mag=mag,
        )
        self._layer = layer

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
            + "Vec3Int.zeros() instead.",
            DeprecationWarning,
        )
        return Vec3Int.zeros()

    @property
    def size(self) -> Vec3Int:
        """⚠️ Deprecated, use `mag_view.bounding_box.in_mag(mag_view.mag).bottomright` instead."""
        warnings.warn(
            "[DEPRECATION] mag_view.size is deprecated. "
            + "Since this is a MagView, please use "
            + "mag_view.bounding_box.in_mag(mag_view.mag).bottomright instead.",
            DeprecationWarning,
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
                + alternative,
                DeprecationWarning,
            )

        if all(i is None for i in [offset, absolute_offset, relative_offset]):
            relative_offset = Vec3Int.zeros()

        mag1_bbox = self._get_mag1_bbox(
            abs_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            current_mag_size=Vec3Int(data.shape[-3:]),
        )

        # Only update the layer's bbox if we are actually larger
        # than the mag-aligned, rounded up bbox (self.bounding_box):
        if not self.bounding_box.contains_bbox(mag1_bbox):
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
    ) -> Iterator[BoundingBox]:
        """
        Returns a Mag(1) bounding box for each file on disk.

        This differs from the bounding box in the properties, which is an "overall" bounding box,
        abstracting from the files on disk.
        """
        for bbox in self._array.list_bounding_boxes():
            yield bbox.from_mag_to_mag1(self._mag)

    def get_views_on_disk(
        self,
        read_only: Optional[bool] = None,
    ) -> Iterator[View]:
        """
        Yields a view for each file on disk, which can be used for efficient parallelization.
        """
        for bbox in self.get_bounding_boxes_on_disk():
            yield self.get_view(
                absolute_offset=bbox.topleft, size=bbox.size, read_only=read_only
            )

    def compress(
        self,
        target_path: Optional[Union[str, Path]] = None,
        args: Optional[Namespace] = None,
    ) -> None:
        """
        Compresses the files on disk. This has consequences for writing data (see `write`).

        The data gets compressed inplace, if target_path is None.
        Otherwise it is written to target_path/layer_name/mag.

        Compressing mags on remote file systems requires a `target_path`.
        """

        from webknossos.dataset.dataset import Dataset

        if target_path is None:
            if self._is_compressed():
                logging.info(f"Mag {self.name} is already compressed")
                return
            else:
                assert is_fs_path(
                    self.path
                ), "Cannot compress a remote mag without `target_path`."
        else:
            target_path = make_upath(target_path)

        uncompressed_full_path = self.path
        compressed_dataset_path = (
            self.layer.dataset.path / f".compress-{uuid4()}"
            if target_path is None
            else target_path
        )
        compressed_dataset = Dataset(
            compressed_dataset_path, scale=self.layer.dataset.scale, exist_ok=True
        )
        compressed_mag = compressed_dataset.get_or_add_layer(
            layer_name=self.layer.name,
            category=self.layer.category,
            dtype_per_channel=self.layer.dtype_per_channel,
            num_channels=self.layer.num_channels,
            data_format=self.layer.data_format,
            largest_segment_id=self.layer._get_largest_segment_id_maybe(),
        ).get_or_add_mag(
            mag=self.mag,
            chunk_size=self.info.chunk_size,
            chunks_per_shard=self.info.chunks_per_shard,
            compress=True,
        )
        compressed_mag.layer.bounding_box = self.layer.bounding_box

        logging.info(
            "Compressing mag {0} in '{1}'".format(
                self.name, str(uncompressed_full_path)
            )
        )
        with get_executor_for_args(args) as executor:
            job_args = []
            for bbox in self.get_bounding_boxes_on_disk():
                bbox = bbox.intersected_with(self.layer.bounding_box, dont_assert=True)
                if not bbox.is_empty():
                    bbox = bbox.align_with_mag(self.mag, ceil=True)
                    source_view = self.get_view(
                        absolute_offset=bbox.topleft, size=bbox.size
                    )
                    target_view = compressed_mag.get_view(
                        absolute_offset=bbox.topleft, size=bbox.size
                    )
                    job_args.append((source_view, target_view))

            wait_and_ensure_success(
                executor.map_to_futures(_compress_cube_job, job_args), "Compressing"
            )

        if target_path is None:
            rmtree(self.path)
            compressed_mag.path.rename(self.path)
            rmtree(compressed_dataset.path)

            # update the handle to the new dataset
            MagView.__init__(
                self,
                self.layer,
                self._mag,
                self.info.chunk_size,
                self.info.chunks_per_shard,
                True,
            )

    @property
    def _properties(self) -> MagViewProperties:
        return next(
            mag_property
            for mag_property in self.layer._properties.mags
            if mag_property.mag == self._mag
        )

    def __repr__(self) -> str:
        return repr(f"MagView(name={self.name}, bounding_box={self.bounding_box})")

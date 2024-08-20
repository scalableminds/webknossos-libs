import logging
import warnings
from argparse import Namespace
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from cluster_tools import Executor
from upath import UPath

from webknossos.dataset.data_format import DataFormat

from ..geometry import Mag, NDBoundingBox, Vec3Int, Vec3IntLike, VecInt
from ..utils import (
    NDArrayLike,
    get_executor_for_args,
    is_fs_path,
    rmtree,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._array import ArrayInfo, BaseArray, WKWArray, ZarrArray, ZarritaArray
from .properties import MagViewProperties

if TYPE_CHECKING:
    from .layer import (
        Layer,
    )

from .view import View


def _find_mag_path_on_disk(
    dataset_path: Path, layer_name: str, mag_name: str, path: Optional[str] = None
) -> Path:
    if path is not None:
        return dataset_path / path

    mag = Mag(mag_name)
    short_mag_file_path = dataset_path / layer_name / mag.to_layer_name()
    long_mag_file_path = dataset_path / layer_name / mag.to_long_layer_name()
    if short_mag_file_path.exists():
        return short_mag_file_path
    else:
        return long_mag_file_path


def _compress_cube_job(args: Tuple[View, View, int]) -> None:
    source_view, target_view, _i = args
    target_view.write(
        source_view.read(), absolute_bounding_box=target_view.bounding_box
    )


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
        chunk_shape: Vec3Int,
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
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            compression_mode=compression_mode,
            axis_order=VecInt(
                0, *layer.bounding_box.index, axes=("c",) + layer.bounding_box.axes
            ),
            shape=VecInt(
                layer.num_channels,
                *VecInt.ones(layer.bounding_box.axes),
                axes=("c",) + layer.bounding_box.axes,
            ),
            dimension_names=("c",) + layer.bounding_box.axes,
        )
        if create:
            self_path = layer.dataset.path / layer.name / mag.to_layer_name()
            BaseArray.get_class(array_info.data_format).create(self_path, array_info)

        super().__init__(
            _find_mag_path_on_disk(layer.dataset.path, layer.name, mag.to_layer_name()),
            array_info,
            bounding_box=layer.bounding_box,
            mag=mag,
        )
        self._layer = layer

    # Overwrites of View methods:
    @property
    def bounding_box(self) -> NDBoundingBox:
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
    def size(self) -> VecInt:
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

    def get_zarr_array(self) -> NDArrayLike:
        """
        Directly access the underlying Zarr array. Only available for Zarr-based datasets.
        """
        array_wrapper = self._array
        if isinstance(array_wrapper, WKWArray):
            raise ValueError("Cannot get the zarr array for wkw datasets.")
        assert isinstance(array_wrapper, ZarrArray) or isinstance(
            array_wrapper, ZarritaArray
        )  # for typechecking
        return array_wrapper._zarray

    def write(
        self,
        data: np.ndarray,
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative, in current mag
        json_update_allowed: bool = True,
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
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

        if all(
            i is None
            for i in [
                offset,
                absolute_offset,
                relative_offset,
                absolute_bounding_box,
                relative_bounding_box,
            ]
        ):
            relative_offset = Vec3Int.zeros()

        if (absolute_bounding_box or relative_bounding_box) is not None:
            data_shape = None
        else:
            data_shape = Vec3Int(data.shape[-3:])

        mag1_bbox = self._get_mag1_bbox(
            abs_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            abs_mag1_bbox=absolute_bounding_box,
            rel_mag1_bbox=relative_bounding_box,
            current_mag_size=data_shape,
        )

        # Only update the layer's bbox if we are actually larger
        # than the mag-aligned, rounded up bbox (self.bounding_box):
        if json_update_allowed and not self.bounding_box.contains_bbox(mag1_bbox):
            self.layer.bounding_box = self.layer.bounding_box.extended_by(mag1_bbox)

        super().write(
            data,
            json_update_allowed=json_update_allowed,
            absolute_bounding_box=mag1_bbox,
        )

    def read(
        self,
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative, in current mag
        size: Optional[
            Vec3IntLike
        ] = None,  # usually in mag1, in current mag if offset is given
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
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
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        read_only: Optional[bool] = None,
    ) -> View:
        # THIS METHOD CAN BE REMOVED WHEN THE DEPRECATED OFFSET IS REMOVED

        # This has other defaults than the View implementation
        # (all deprecations are handled in the superclass)
        bb = self.bounding_box.in_mag(self._mag)
        if offset is not None and size is None:
            offset = Vec3Int(offset)
            size = bb.bottomright - offset

        return super().get_view(
            None if offset is None else Vec3Int(offset) - bb.topleft,
            size,
            relative_offset=relative_offset,
            absolute_offset=absolute_offset,
            relative_bounding_box=relative_bounding_box,
            absolute_bounding_box=absolute_bounding_box,
            read_only=read_only,
        )

    def get_bounding_boxes_on_disk(
        self,
    ) -> Iterator[NDBoundingBox]:
        """
        Returns a Mag(1) bounding box for each file on disk.

        This differs from the bounding box in the properties, which is an "overall" bounding box,
        abstracting from the files on disk.
        """
        try:
            bboxes = self._array.list_bounding_boxes()
        except NotImplementedError:
            warnings.warn(
                "[WARNING] The underlying array storage does not support listing the stored bounding boxes. "
                + "Instead all bounding boxes are iterated, which can be slow."
            )
            bboxes = self.bounding_box.in_mag(self.mag).chunk(
                self._array.info.shard_shape
            )
        for bbox in bboxes:
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
        args: Optional[Namespace] = None,  # deprecated
        executor: Optional[Executor] = None,
    ) -> None:
        """
        Compresses the files on disk. This has consequences for writing data (see `write`).

        The data gets compressed inplace, if target_path is None.
        Otherwise it is written to target_path/layer_name/mag.

        Compressing mags on remote file systems requires a `target_path`.
        """

        from .dataset import Dataset

        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        if target_path is None:
            if self._is_compressed():
                logging.info(f"Mag {self.name} is already compressed")
                return
            else:
                assert is_fs_path(
                    self.path
                ), "Cannot compress a remote mag without `target_path`."
        else:
            target_path = UPath(target_path)

        uncompressed_full_path = self.path
        compressed_dataset_path = (
            self.layer.dataset.path / f".compress-{uuid4()}"
            if target_path is None
            else target_path
        )
        compressed_dataset = Dataset(
            compressed_dataset_path,
            voxel_size=self.layer.dataset.voxel_size,
            exist_ok=True,
        )
        compressed_layer = compressed_dataset.get_or_add_layer(
            layer_name=self.layer.name,
            category=self.layer.category,
            dtype_per_channel=self.layer.dtype_per_channel,
            num_channels=self.layer.num_channels,
            data_format=self.layer.data_format,
            largest_segment_id=self.layer._get_largest_segment_id_maybe(),
        )
        compressed_layer.bounding_box = self.layer.bounding_box
        compressed_mag = compressed_layer.get_or_add_mag(
            mag=self.mag,
            chunk_shape=self.info.chunk_shape,
            chunks_per_shard=self.info.chunks_per_shard,
            compress=True,
        )

        logging.info(
            "Compressing mag {0} in '{1}'".format(
                self.name, str(uncompressed_full_path)
            )
        )
        with get_executor_for_args(args, executor) as executor:
            if self.layer.data_format == DataFormat.WKW:
                job_args = []
                for i, bbox in enumerate(self._array.list_bounding_boxes()):
                    bbox = bbox.from_mag_to_mag1(self._mag).intersected_with(
                        self.layer.bounding_box, dont_assert=True
                    )
                    if not bbox.is_empty():
                        bbox = bbox.align_with_mag(self.mag, ceil=True)
                        source_view = self.get_view(
                            absolute_offset=bbox.topleft, size=bbox.size
                        )
                        target_view = compressed_mag.get_view(
                            absolute_offset=bbox.topleft, size=bbox.size
                        )
                        job_args.append((source_view, target_view, i))

                wait_and_ensure_success(
                    executor.map_to_futures(_compress_cube_job, job_args),
                    executor=executor,
                    progress_desc=f"Compressing {self.layer.name} {self.name}",
                )
            else:
                warnings.warn(
                    "[WARNING] The underlying array storage does not support listing the stored bounding boxes. "
                    + "Instead all bounding boxes are iterated, which can be slow."
                )
                self.for_zipped_chunks(
                    target_view=compressed_mag,
                    executor=executor,
                    func_per_chunk=_compress_cube_job,
                    progress_desc=f"Compressing {self.layer.name} {self.name}",
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
                self.info.chunk_shape,
                self.info.chunks_per_shard,
                True,
            )

    def merge_with_view(
        self,
        other: "MagView",
        executor: Executor,
    ) -> None:
        assert all(
            other.info.chunks_per_shard.to_np() == 1
        ), "volume annotation must have file_len=1"
        assert (
            self.info.voxel_type == other.info.voxel_type
        ), "Volume annotation must have same dtype as fallback layer"
        assert (
            self.mag == other.mag
        ), f"To merge two Views, both need the same mag: Own mag {self.mag} does not match other mag {other.mag}"

        logging.info("Scan disk for annotation shards.")
        bboxes = list(bbox for bbox in other.get_bounding_boxes_on_disk())

        logging.info("Grouping %s bboxes according to output shards.", len(bboxes))
        shards_with_bboxes = NDBoundingBox.group_boxes_with_aligned_mag(
            bboxes, Mag(self.info.shard_shape * self.mag)
        )

        args = [(other, shard, bboxes) for shard, bboxes in shards_with_bboxes.items()]

        logging.info("Merging %s shards.", len(args))
        wait_and_ensure_success(
            executor.map_to_futures(self.merge_chunk, args),
            executor,
            "Merging chunks with fallback layer",
        )

    def merge_chunk(
        self, args: Tuple["MagView", NDBoundingBox, List[NDBoundingBox]]
    ) -> None:
        other, shard, bboxes = args
        data_buffer = self.read(absolute_bounding_box=shard)[0]

        for bbox in bboxes:
            read_data = other.read(absolute_bounding_box=bbox)[0]
            data_buffer[bbox.offset(-shard.topleft).in_mag(other.mag).to_slices()] = (
                read_data
            )

        self.write(data_buffer, absolute_offset=shard.topleft)

    @property
    def _properties(self) -> MagViewProperties:
        return next(
            mag_property
            for mag_property in self.layer._properties.mags
            if mag_property.mag == self._mag
        )

    def __repr__(self) -> str:
        return f"MagView(name={repr(self.name)}, bounding_box={self.bounding_box})"

    @classmethod
    def _ensure_mag_view(cls, mag_view: Union[str, PathLike, "MagView"]) -> "MagView":
        if isinstance(mag_view, MagView):
            return mag_view
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            mag_view_path = UPath(mag_view)
            return (
                Dataset.open(mag_view_path.parent.parent)
                .get_layer(mag_view_path.parent.name)
                .get_mag(mag_view_path.name)
            )

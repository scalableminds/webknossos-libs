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
    LazyPath,
    get_executor_for_args,
    is_fs_path,
    rmtree,
    strip_trailing_slash,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._array import ArrayInfo, BaseArray, TensorStoreArray, WKWArray
from .properties import MagViewProperties

if TYPE_CHECKING:
    import tensorstore

    from .layer import (
        Layer,
    )

from .view import View


def _find_mag_path(
    dataset_path: Path,
    layer_name: str,
    mag_name: str,
    path: Optional[LazyPath] = None,
) -> LazyPath:
    if path is not None:
        return path

    mag = Mag(mag_name)
    short_mag_file_path = dataset_path / layer_name / mag.to_layer_name()
    long_mag_file_path = dataset_path / layer_name / mag.to_long_layer_name()
    return LazyPath(short_mag_file_path, long_mag_file_path)


def _compress_cube_job(args: Tuple[View, View, int]) -> None:
    source_view, target_view, _i = args
    target_view.write(
        source_view.read(), absolute_bounding_box=target_view.bounding_box
    )


class MagView(View):
    """A view of a specific magnification level within a WEBKNOSSOS layer.

    MagView provides access to volumetric data at a specific resolution/magnification level.
    It supports reading, writing, and processing operations on the underlying data, with
    coordinates automatically handled in the correct magnification space.

    Key Features:
        - Read/write volumetric data at specific magnification levels
        - Automatic coordinate transformation between Mag(1) and current magnification
        - Support for compressed and uncompressed data formats
        - Chunked processing for efficient memory usage
        - Downsampling and upsampling capabilities

    Attributes:
        layer: The parent Layer object this magnification belongs to.
        mag: The magnification level (e.g., Mag(1), Mag(2), Mag(4), etc.).
        info: Information about array storage (chunk shape, compression, etc.).
        path: Path to the data on disk.
        bounding_box: The spatial extent of this magnification in Mag(1) coordinates.

    Examples:
        ```python
        # Create a dataset with a segmentation layer
        ds = Dataset("path/to/dataset", voxel_size=(1, 1, 1))
        layer = ds.add_layer("segmentation", SEGMENTATION_CATEGORY)

        # Add and work with magnification levels
        mag1 = layer.add_mag(Mag(1))
        mag2 = layer.add_mag(Mag(2))

        # Write data at Mag(1)
        mag1.write(data, absolute_offset=(100, 200, 300))

        # Read data at Mag(2) - coordinates are in Mag(1) space
        data = mag2.read(absolute_offset=(100, 200, 300), size=(512, 512, 512))

        # Process data in chunks
        def process_chunk(view: View) -> None:
            data = view.read()
            # Process data...
            view.write(processed_data)

        mag1.for_each_chunk(process_chunk)
        ```

    Notes:
        - All offset/size parameters in read/write methods expect Mag(1) coordinates
        - Use get_view() to obtain restricted views of the data
        - Compressed data operations may have performance implications
        - When writing to segmentation layers, update largest_segment_id as needed

    See Also:
        - Layer: Parent container for magnification levels
        - View: Base class providing data access methods
        - Dataset: Root container for all layers
    """

    @classmethod
    def create(
        cls,
        layer: "Layer",
        mag: Mag,
        chunk_shape: Vec3Int,
        chunks_per_shard: Vec3Int,
        compression_mode: bool,
        path: Optional[LazyPath] = None,
    ) -> "MagView":
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.add_mag()`.
        """
        array_info = ArrayInfo(
            data_format=layer._properties.data_format,
            voxel_type=layer.dtype_per_channel,
            num_channels=layer.num_channels,
            chunk_shape=chunk_shape,
            shard_shape=chunk_shape * chunks_per_shard,
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
        mag_path = (
            path.resolve()
            if path
            else layer.dataset.path / layer.name / mag.to_layer_name()
        )
        BaseArray.get_class(array_info.data_format).create(mag_path, array_info)
        return cls(layer, mag, LazyPath.resolved(mag_path))

    def __init__(
        self,
        layer: "Layer",
        mag: Mag,
        mag_path: LazyPath,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.get_mag()`.
        """

        super().__init__(
            mag_path,
            bounding_box=layer.bounding_box,
            mag=mag,
            data_format=layer.data_format,
        )
        self._layer = layer

    # Overwrites of View methods:
    @property
    def bounding_box(self) -> NDBoundingBox:
        """Get the spatial extent of this magnification level in Mag(1) coordinates.

        Returns:
            NDBoundingBox: The bounding box of this magnification level in Mag(1) coordinates.

        Notes:
            - The bounding box is automatically aligned with the magnification level
            - It represents the overall extent of the data, potentially including empty regions
        """
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
        """Get the parent Layer object.

        Returns:
            Layer: The Layer object that contains this magnification level.

        Notes:
            - The Layer provides context about data type, category, and overall properties
            - Used internally for coordinate transformations and data validation
        """
        return self._layer

    @property
    def path(self) -> Path:
        """Get the path to this magnification level's data.

        Returns:
            Path: Path to the data files on disk.

        Notes:
            - Path may be local or remote depending on dataset configuration
        """
        return self._path.resolve()

    @property
    def is_remote_to_dataset(self) -> bool:
        """Check if this magnification's data is stored remotely on a server relative to the dataset.

        Returns:
            bool: True if data is stored in a different location than the parent dataset.

        """
        return self._path.resolve().parent.parent != self.layer.dataset.path

    @property
    def name(self) -> str:
        """Get the name of this magnification level.

        Returns:
            str: String representation of the magnification level (e.g., "1-1-1" for Mag(1)).

        """
        return self._mag.to_layer_name()

    def get_zarr_array(self) -> "tensorstore.TensorStore":
        """Get direct access to the underlying Zarr array.

        Provides direct access to the underlying Zarr array for advanced operations.
        Only available for Zarr-based datasets.

        Returns:
            NDArrayLike: The underlying Zarr array object.

        Raises:
            ValueError: If called on a non-Zarr dataset.

        Notes:
            - Only works with Zarr-based datasets
            - Provides low-level access to data storage
            - Use with caution as it bypasses normal access patterns
        """
        array_wrapper = self._array
        if isinstance(array_wrapper, WKWArray):
            raise ValueError("Cannot get the zarr array for wkw datasets.")
        assert isinstance(
            array_wrapper, TensorStoreArray
        ), f"Expected TensorStoreArray, got {type(array_wrapper)}"  # for typechecking
        return array_wrapper._array

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
        """Write volumetric data to the magnification level.

        This method writes numpy array data to the dataset at the specified location. All offset and bounding box
        coordinates are expected to be in Mag(1) space, regardless of the current magnification level.

        Args:
            data: Numpy array containing the volumetric data to write. Shape must match the target region.
            offset: ⚠️ Deprecated. Use relative_offset or absolute_offset instead.
            json_update_allowed: If True, allows updating the layer's bounding box if the write extends beyond it.
            relative_offset: Optional offset relative to the view's position in Mag(1) coordinates.
            absolute_offset: Optional absolute position in Mag(1) coordinates.
            relative_bounding_box: Optional bounding box relative to view's position in Mag(1) coordinates.
            absolute_bounding_box: Optional absolute bounding box in Mag(1) coordinates.

        Examples:
            ```python
            # Write data at absolute position
            mag1.write(data, absolute_offset=(100, 200, 300))

            # Write using bounding box
            bbox = BoundingBox((0, 0, 0), (100, 100, 100))
            mag2.write(data, absolute_bounding_box=bbox)
            ```

        Notes:
            - At least one of offset/bounding_box parameters must be provided
            - Data shape must match the target region size
            - Coordinates are automatically scaled based on magnification
            - For compressed data, writing may be slower due to compression
            - Large writes may temporarily increase memory usage
        """
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
        """Read volumetric data from the magnification level.

        This method reads data from the dataset at the specified location. All offset and bounding box
        coordinates are expected to be in Mag(1) space, regardless of the current magnification level.

        Args:
            offset: ⚠️ Deprecated. Use relative_offset or absolute_offset instead.
            size: Size of region to read. In Mag(1) coordinates unless used with deprecated offset.
            relative_offset: Optional offset relative to the view's position in Mag(1) coordinates.
            absolute_offset: Optional absolute position in Mag(1) coordinates.
            relative_bounding_box: Optional bounding box relative to view's position in Mag(1) coordinates.
            absolute_bounding_box: Optional absolute bounding box in Mag(1) coordinates.

        Returns:
            np.ndarray: The volumetric data as a numpy array.

        Examples:
            ```python
            # Read data at absolute position
            data = mag1.read(absolute_offset=(100, 200, 300), size=(512, 512, 512))

            # Read using bounding box
            bbox = BoundingBox((0, 0, 0), (100, 100, 100))
            data = mag2.read(absolute_bounding_box=bbox)
            ```

        Notes:
            - At least one of offset/bounding_box parameters must be provided
            - Coordinates are automatically scaled based on magnification
            - For compressed data, reading includes decompression time
            - Large reads may temporarily increase memory usage
        """
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
        """Get a restricted view of this magnification level.

        Creates a new View object that represents a subset of this magnification level's data.
        The view can be used to read/write data within its bounds. All offset and bounding box
        coordinates are expected to be in Mag(1) space, regardless of the current magnification level.

        Args:
            offset: ⚠️ Deprecated. Use relative_offset or absolute_offset instead.
            size: Size of region to view. In Mag(1) coordinates unless used with deprecated offset.
            relative_offset: Optional offset relative to the view's position in Mag(1) coordinates.
            absolute_offset: Optional absolute position in Mag(1) coordinates.
            relative_bounding_box: Optional bounding box relative to view's position in Mag(1) coordinates.
            absolute_bounding_box: Optional absolute bounding box in Mag(1) coordinates.
            read_only: If True, the view will be read-only. If None, determined by context.

        Returns:
            View: A new View object representing the specified region.

        Examples:
            ```python
            # Get view at absolute position
            view = mag1.get_view(absolute_offset=(100, 200, 300), size=(512, 512, 512))
            data = view.read()  # Read from the view
            view.write(data)    # Write to the view

            # Get view using bounding box
            bbox = BoundingBox((0, 0, 0), (100, 100, 100))
            view = mag2.get_view(absolute_bounding_box=bbox)
            ```

        Notes:
            - Views are lightweight objects that don't copy data
            - Read-only views prevent accidental data modification
            - Views can be used for efficient parallel processing
            - Coordinates are automatically scaled based on magnification
        """
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
        """Returns a Mag(1) bounding box for each file on disk.

        This method iterates through the actual files stored on disk and returns their bounding boxes.
        This is different from the layer's bounding box property, which represents the overall extent
        of the data, potentially including regions without actual data files.

        Returns:
            Iterator[NDBoundingBox]: Iterator yielding bounding boxes in Mag(1) coordinates.

        Examples:
            ```python
            # Print all data file bounding boxes
            for bbox in mag1.get_bounding_boxes_on_disk():
                print(f"Found data file at {bbox}")

            # Calculate total data volume
            total_volume = sum(bbox.volume() for bbox in mag1.get_bounding_boxes_on_disk())
            ```

        Notes:
            - Bounding boxes are in Mag(1) coordinates
            - Some storage formats may not support efficient listing
            - For unsupported formats, falls back to chunk-based iteration
            - Useful for understanding actual data distribution on disk
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
        """Yields a view for each file on disk for efficient parallelization.

        Creates View objects that correspond to actual data files on disk. This is particularly
        useful for parallel processing as each view represents a natural unit of data storage.

        Args:
            read_only: If True, returned views will be read-only. If None, determined by context.

        Returns:
            Iterator[View]: Iterator yielding View objects for each data file.

        Examples:
            ```python
            # Process each data file in parallel
            def process_chunk(view: View) -> None:
                data = view.read()
                # Process data...
                if not view.read_only:
                    view.write(processed_data)

            with get_executor_for_args(None) as executor:
                executor.map(process_chunk, mag1.get_views_on_disk())
            ```

        Notes:
            - Views correspond to actual files/chunks on disk
            - Ideal for parallel processing of large datasets
            - Each view's bounding box aligns with storage boundaries
            - Memory efficient as only one chunk is loaded at a time
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
        """Compresses the files on disk.

        Compresses the magnification level's data, either in-place or to a new location.
        Compression can reduce storage space but may impact read/write performance.

        Args:
            target_path: Optional path to write compressed data. If None, compresses in-place.
            args: ⚠️ Deprecated. Use executor parameter instead.
            executor: Optional executor for parallel compression.

        Examples:
            ```python
            # Compress in-place
            mag1.compress()
            ```

        Notes:
            - In-place compression requires local filesystem
            - Remote compression requires target_path
            - Compression is parallelized when executor is provided
            - Progress is displayed during compression
            - Compressed data may have slower read/write speeds
        """
        from .dataset import Dataset

        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        path = self._path.resolve()
        if target_path is None:
            if self._is_compressed():
                logging.info(f"Mag {self.name} is already compressed")
                return
            else:
                assert is_fs_path(
                    path
                ), "Cannot compress a remote mag without `target_path`."
        else:
            target_path = UPath(target_path)

        uncompressed_full_path = path
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
            rmtree(path)
            compressed_mag.path.rename(path)
            rmtree(compressed_dataset.path)

            # update the handle to the new dataset
            MagView.__init__(self, self.layer, self._mag, LazyPath.resolved(path))

    def merge_with_view(
        self,
        other: "MagView",
        executor: Executor,
    ) -> None:
        """Merges data from another view into this one.

        Combines data from another MagView into this one, using this view's data as the base
        and overlaying the other view's data where present. This is particularly useful for
        merging annotations or overlays.

        Args:
            other: The MagView to merge into this one.
            executor: Executor for parallel merging operations.

        Notes:
            - Both views must have same magnification
            - Other view must have file_len = 1
            - Both views must have same voxel type
            - Merging is parallelized using the provided executor
            - Updates layer bounding box if necessary
        """
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

        new_bbox = self.bounding_box
        for shard_bbox in shards_with_bboxes.keys():
            new_bbox = new_bbox.extended_by(shard_bbox)

        logging.info(f"Set mag layer bounding box to {new_bbox}")
        self.layer.bounding_box = new_bbox

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
        """Merge a single chunk during parallel merge operations.

        Internal method used by merge_with_view() for parallel processing. Merges data
        from another view into this one for a specific chunk region.

        Args:
            args: Tuple containing:
                - other (MagView): Source view to merge from
                - shard (NDBoundingBox): Target shard region
                - bboxes (List[NDBoundingBox]): List of source bounding boxes
        """
        other, shard, bboxes = args
        data_buffer = self.read(absolute_bounding_box=shard)[0]

        for bbox in bboxes:
            read_data = other.read(absolute_bounding_box=bbox)[0]
            data_buffer[bbox.offset(-shard.topleft).in_mag(other.mag).to_slices()] = (
                read_data
            )

        self.write(data_buffer, absolute_offset=shard.topleft)

    @classmethod
    def _ensure_mag_view(cls, mag_view: Union[str, PathLike, "MagView"]) -> "MagView":
        """Ensure input is a MagView object, converting path-like objects if needed.

        Internal helper method that converts various input types into a MagView object.
        If the input is already a MagView, returns it directly. Otherwise, attempts to
        create a MagView from the provided path.

        Args:
            mag_view: Input that should be converted to a MagView. Can be:
                - MagView object: returned as-is
                - str or PathLike: path to a magnification level

        Returns:
            MagView: A valid MagView object.
        """
        if isinstance(mag_view, MagView):
            return mag_view
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            path = UPath(
                str(mag_view.path) if isinstance(mag_view, MagView) else str(mag_view)
            )
            mag_view_path = strip_trailing_slash(path)
            return (
                Dataset.open(mag_view_path.parent.parent)
                .get_layer(mag_view_path.parent.name)
                .get_mag(mag_view_path.name)
            )

    @property
    def _properties(self) -> MagViewProperties:
        """Get the properties object for this magnification level.

        Internal method that retrieves the properties object containing configuration
        and metadata for this magnification level.

        Returns:
            MagViewProperties: Properties object for this magnification level.

        """
        return next(
            mag_property
            for mag_property in self.layer._properties.mags
            if mag_property.mag == self._mag
        )

    def __repr__(self) -> str:
        return f"MagView(name={repr(self.name)}, bounding_box={self.bounding_box})"

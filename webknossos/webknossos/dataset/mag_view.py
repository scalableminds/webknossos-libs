import logging
import warnings
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Union
from uuid import uuid4

import numpy as np
from cluster_tools import Executor
from upath import UPath

from ..geometry import Mag, NDBoundingBox, Vec3Int, Vec3IntLike, VecInt
from ..utils import (
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

logger = logging.getLogger(__name__)


def _compress_cube_job(args: tuple[View, View, int]) -> None:
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
        read_only: Whether this magnification is read-only.

    Examples:
        ```python
        # Create a dataset with a segmentation layer
        ds = Dataset("path/to/dataset", voxel_size=(1, 1, 1))
        layer = ds.add_layer(
            "segmentation",
            SEGMENTATION_CATEGORY,
            bounding_box=BoundingBox((100, 200, 300), (512, 512, 512)),
        )

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

    _layer: "Layer"

    @classmethod
    def create(
        cls,
        layer: "Layer",
        mag: Mag,
        *,
        path: UPath,
        chunk_shape: Vec3Int,
        shard_shape: Vec3Int | None = None,
        chunks_per_shard: Vec3Int | None = None,
        compression_mode: bool,
        read_only: bool = False,
    ) -> "MagView":
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.add_mag()`.
        """
        if chunks_per_shard is None:
            if shard_shape is None:
                raise ValueError("Please provide shard_shape or chunks_per_shard.")
        else:
            if shard_shape is not None:
                raise ValueError(
                    "shard_shape or chunks_per_shard must not be provided at the same time."
                )
            else:
                warn_deprecated("chunks_per_shard", "shard_shape")
                shard_shape = chunk_shape * chunks_per_shard

        array_info = ArrayInfo(
            data_format=layer._properties.data_format,
            voxel_type=layer.dtype_per_channel,
            num_channels=layer.num_channels,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
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

        BaseArray.get_class(array_info.data_format).create(path, array_info)
        return cls(layer, mag, path, read_only=read_only)

    def __init__(
        self,
        layer: "Layer",
        mag: Mag,
        path: UPath,
        read_only: bool = False,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.layer.Layer.get_mag()`.
        """

        super().__init__(
            path,
            bounding_box=layer.bounding_box,
            mag=mag,
            data_format=layer.data_format,
            read_only=read_only,
        )
        self._layer = layer

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise RuntimeError(f"{self} is read-only, the changes will not be saved!")

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
    def path(self) -> UPath:
        """Get the path to this magnification level's data.

        Returns:
            UPath: Path to the data files on disk.

        Notes:
            - Path may be local or remote depending on dataset configuration
        """
        return self._path

    @property
    def is_foreign(self) -> bool:
        """Check if this magnification's data is stored not as part of to the dataset.
        Returns:
            bool: True if data is stored in a different location than the parent dataset.
        """
        return self.path.parent.parent != self.layer.dataset.resolved_path

    @property
    def is_remote_to_dataset(self) -> bool:
        warn_deprecated("is_remote_to_dataset", "is_foreign")
        return self.is_foreign

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
        assert isinstance(array_wrapper, TensorStoreArray), (
            f"Expected TensorStoreArray, got {type(array_wrapper)}"
        )  # for typechecking
        return array_wrapper._array

    def write(
        self,
        data: np.ndarray,
        *,
        allow_resize: bool = False,
        allow_unaligned: bool = False,
        relative_offset: Vec3IntLike | None = None,  # in mag1
        absolute_offset: Vec3IntLike | None = None,  # in mag1
        relative_bounding_box: NDBoundingBox | None = None,  # in mag1
        absolute_bounding_box: NDBoundingBox | None = None,  # in mag1
    ) -> None:
        """Write volumetric data to the magnification level.

        This method writes numpy array data to the dataset at the specified location. All offset and bounding box
        coordinates are expected to be in Mag(1) space, regardless of the current magnification level.

        Args:
            data (np.ndarray): The data to write. For 3D data, shape should
                be (x, y, z). For multi-channel 3D data, shape should be (channels, x, y, z).
                For n-dimensional data, the axes must match the bounding box axes of the layer.
                Shape must match the target region size.
            allow_resize: If True, allows updating the layer's bounding box if the write
                extends beyond it.
                Must not be set to True, when writing to the same magnification level in parallel.
                For one-off writes, consider using `Dataset.write_layer`.
                Defaults to False.
            allow_unaligned (bool, optional): If True, allows writing data to without
                being aligned to the shard shape.
                Defaults to False.
            relative_offset (Vec3IntLike | None, optional): Offset relative to view's
                position in Mag(1) coordinates. Defaults to None.
            absolute_offset (Vec3IntLike | None, optional): Absolute offset in Mag(1)
                coordinates. Defaults to None.
            relative_bounding_box (NDBoundingBox | None, optional): Bounding box relative
                to view's position in Mag(1) coordinates. Defaults to None.
            absolute_bounding_box (NDBoundingBox | None, optional): Absolute bounding box
                in Mag(1) coordinates. Defaults to None.

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
        self._ensure_writable()
        if all(
            i is None
            for i in [
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
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            abs_mag1_bbox=absolute_bounding_box,
            rel_mag1_bbox=relative_bounding_box,
            current_mag_size=data_shape,
        )

        # Only update the layer's bbox if we are actually larger
        # than the mag-aligned, rounded up bbox (self.bounding_box):
        if not self.bounding_box.contains_bbox(mag1_bbox):
            if allow_resize:
                self.layer.bounding_box = self.layer.bounding_box.extended_by(mag1_bbox)
            else:
                raise ValueError(
                    f"The bounding box to write {mag1_bbox} does not fit in the layer's bounding box {self.layer.bounding_box}. "
                    + "Please use `allow_resize=True` or explicitly resize the bounding box beforehand."
                )

        super().write(
            data,
            allow_unaligned=allow_unaligned,
            absolute_bounding_box=mag1_bbox,
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
        read_only: bool | None = None,
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
        *,
        target_path: str | Path | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Compresses the files on disk.

        Compresses the magnification level's data, either in-place or to a new location.
        Compression can reduce storage space but may impact read/write performance.

        Args:
            target_path: Optional path to write compressed data. If None, compresses in-place.
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

        self._ensure_writable()
        path = self._path
        if target_path is None:
            if self._is_compressed():
                logger.info(f"Mag {self.name} is already compressed")
                return
            else:
                assert is_fs_path(path), (
                    "Cannot compress a remote mag without `target_path`."
                )
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
            shard_shape=self.info.shard_shape,
            compress=True,
        )

        logger.info(f"Compressing mag {self.name} in '{str(uncompressed_full_path)}'")
        with get_executor_for_args(None, executor) as executor:
            try:
                bbox_iterator = self._array.list_bounding_boxes()
            except NotImplementedError:
                bbox_iterator = None
            if bbox_iterator is not None:
                job_args = []
                for i, bbox in enumerate(bbox_iterator):
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
            MagView.__init__(self, self.layer, self._mag, path)

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
        self._ensure_writable()
        assert all(other.info.chunks_per_shard.to_np() == 1), (
            "volume annotation must have file_len=1"
        )
        assert self.info.voxel_type == other.info.voxel_type, (
            "Volume annotation must have same dtype as fallback layer"
        )
        assert self.mag == other.mag, (
            f"To merge two Views, both need the same mag: Own mag {self.mag} does not match other mag {other.mag}"
        )

        logger.info("Scan disk for annotation shards.")
        bboxes = list(bbox for bbox in other.get_bounding_boxes_on_disk())

        logger.info("Grouping %s bboxes according to output shards.", len(bboxes))
        shards_with_bboxes = NDBoundingBox.group_boxes_with_aligned_mag(
            bboxes, Mag(self.info.shard_shape * self.mag)
        )

        new_bbox = self.bounding_box
        for shard_bbox in shards_with_bboxes.keys():
            new_bbox = new_bbox.extended_by(shard_bbox)

        logger.info(f"Set mag layer bounding box to {new_bbox}")
        self.layer.bounding_box = new_bbox

        args = [(other, shard, bboxes) for shard, bboxes in shards_with_bboxes.items()]

        logger.info("Merging %s shards.", len(args))
        wait_and_ensure_success(
            executor.map_to_futures(self.merge_chunk, args),
            executor,
            "Merging chunks with fallback layer",
        )

    def merge_chunk(
        self, args: tuple["MagView", NDBoundingBox, list[NDBoundingBox]]
    ) -> None:
        """Merge a single chunk during parallel merge operations.

        Internal method used by merge_with_view() for parallel processing. Merges data
        from another view into this one for a specific chunk region.

        Args:
            args: Tuple containing:
                - other (MagView): Source view to merge from
                - shard (NDBoundingBox): Target shard region
                - bboxes (list[NDBoundingBox]): List of source bounding boxes
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
    def _ensure_mag_view(
        cls, mag_view_or_path: Union[str, PathLike, "MagView"]
    ) -> "MagView":
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
        if isinstance(mag_view_or_path, MagView):
            return mag_view_or_path
        else:
            # local import to prevent circular dependency
            from .dataset import Dataset

            mag_view_path = strip_trailing_slash(UPath(mag_view_or_path))
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

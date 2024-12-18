import warnings
from argparse import Namespace
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
import wkw
from cluster_tools import Executor

from ..geometry import BoundingBox, Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from ..geometry.vec_int import VecInt, VecIntLike
from ..utils import (
    LazyPath,
    count_defined_values,
    get_executor_for_args,
    get_rich_progress,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._array import ArrayInfo, BaseArray, WKWArray
from .data_format import DataFormat

if TYPE_CHECKING:
    from ._utils.buffered_slice_reader import BufferedSliceReader
    from ._utils.buffered_slice_writer import BufferedSliceWriter


def _assert_check_equality(args: Tuple["View", "View", int]) -> None:
    view_a, view_b, _ = args
    assert np.all(view_a.read() == view_b.read())


_BLOCK_ALIGNMENT_WARNING = (
    "[WARNING] write() was called on a compressed mag without block alignment. "
    + "Performance will be degraded as the data has to be padded first."
)


class View:
    """A View represents a bounding box to a region of a specific StorageBackend with additional functionality.

    The View class provides a way to access and manipulate a specific region of data within a dataset.
    Write operations are restricted to the defined bounding box. Views are designed to be easily passed
    around as parameters and can be used to efficiently work with subsets of larger datasets.

    Examples:
        ```python
        from webknossos.dataset import Dataset, View
        dataset = Dataset.open("path/to/dataset")

        # Get a view for a specific layer at mag 1
        layer = dataset.get_layer("color")
        view = layer.get_mag("1").get_view(size=(100, 100, 10))

        # Read data from the view
        data = view.read()

        # Write data to the view (if not read_only)
        import numpy as np
        view.write(np.zeros(view.bounding_box.in_mag(view.mag).size))
        ```
    """

    _path: LazyPath
    _data_format: DataFormat
    _bounding_box: Optional[NDBoundingBox]
    _read_only: bool
    _cached_array: Optional[BaseArray]
    _mag: Mag

    def __init__(
        self,
        path_to_mag_view: LazyPath,
        bounding_box: Optional[
            NDBoundingBox
        ],  # in mag 1, absolute coordinates, optional only for mag_view since it overwrites the bounding_box property
        mag: Mag,
        data_format: DataFormat,
        read_only: bool = False,
    ):
        """Initialize a View instance for accessing and manipulating dataset regions.

        Note: Do not use this constructor manually. Instead use `View.get_view()`
        (also available on a `MagView`) to get a `View`.

        Args:
            path_to_mag_view (Path): Path to the magnification view directory.
            array_info (ArrayInfo): Information about the array structure and properties.
            bounding_box (Optional[NDBoundingBox]): The bounding box in mag 1 absolute coordinates.
                Optional only for mag_view since it overwrites the bounding_box property.
            mag (Mag): Magnification level of the view.
            read_only (bool, optional): Whether the view is read-only. Defaults to False.

        Examples:
            ```python
            # The recommended way to create a View is through get_view():
            layer = dataset.get_layer("color")
            mag_view = layer.get_mag("1")
            view = mag_view.get_view(size=(100, 100, 10))
            ```
        """
        self._path = path_to_mag_view
        self._data_format = data_format
        self._bounding_box = bounding_box
        self._read_only = read_only
        self._cached_array = None
        self._mag = mag

    @property
    def info(self) -> ArrayInfo:
        """Get information about the array structure and properties.

        Returns:
            ArrayInfo: Object containing array metadata such as data type,
                num_channels, and other array-specific information.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            array_info = view.info
            print(f"Data type: {array_info.data_type}")
            print(f"Num channels: {array_info.num_channels}")
            ```
        """
        return self._array.info

    @property
    def header(self) -> wkw.Header:
        """⚠️ Deprecated, use `info` instead."""
        warn_deprecated("header", "info")
        assert isinstance(
            self._array, WKWArray
        ), "`header` only works with WKW datasets."
        return self._array._wkw_dataset.header

    @property
    def bounding_box(self) -> NDBoundingBox:
        """Gets the bounding box of this view.

        The bounding box defines the region of interest within the dataset in absolute
        coordinates at magnification level 1. It specifies both the position and size
        of the view's data.

        Returns:
            NDBoundingBox: A bounding box object representing the view's boundaries
                in absolute coordinates at magnification level 1.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            bbox = view.bounding_box
            print(f"Top-left corner: {bbox.topleft}")
            print(f"Size: {bbox.size}")
            ```
        """
        assert self._bounding_box is not None
        return self._bounding_box

    @property
    def mag(self) -> Mag:
        """Gets the magnification level of this view.

        The magnification level determines the resolution at which the data is accessed.
        A higher magnification number means lower resolution (e.g., mag 2 means every
        second voxel, mag 4 means every fourth voxel, etc.).

        Returns:
            Mag: The magnification level of this view.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            print(f"Current magnification: {view.mag}")
            ```
        """
        return self._mag

    @property
    def read_only(self) -> bool:
        """Indicates whether this view is read-only.

        When a view is read-only, write operations are not permitted. This property
        helps prevent accidental modifications to the dataset.

        Returns:
            bool: True if the view is read-only, False if write operations are allowed.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            if not view.read_only:
                view.write(data)
            else:
                print("Cannot modify read-only view")
            ```
        """
        return self._read_only

    @property
    def global_offset(self) -> VecInt:
        """⚠️ Deprecated, use `view.bounding_box.in_mag(view.mag).topleft` instead."""
        warnings.warn(
            "[DEPRECATION] view.global_offset is deprecated. "
            + "Since this is a View, please use "
            + "view.bounding_box.in_mag(view.mag).topleft instead.",
            DeprecationWarning,
        )
        return self.bounding_box.in_mag(self._mag).topleft

    @property
    def size(self) -> VecInt:
        """⚠️ Deprecated, use `view.bounding_box.in_mag(view.mag).size` instead."""
        warnings.warn(
            "[DEPRECATION] view.size is deprecated. "
            + "Since this is a View, please use "
            + "view.bounding_box.in_mag(view.mag).size instead.",
            DeprecationWarning,
        )
        return self.bounding_box.in_mag(self._mag).size

    def _get_mag1_bbox(
        self,
        abs_mag1_bbox: Optional[NDBoundingBox] = None,
        rel_mag1_bbox: Optional[NDBoundingBox] = None,
        abs_mag1_offset: Optional[Vec3IntLike] = None,
        rel_mag1_offset: Optional[Vec3IntLike] = None,
        mag1_size: Optional[Vec3IntLike] = None,
        abs_current_mag_offset: Optional[Vec3IntLike] = None,
        rel_current_mag_offset: Optional[Vec3IntLike] = None,
        current_mag_size: Optional[Vec3IntLike] = None,
    ) -> NDBoundingBox:
        num_bboxes = count_defined_values([abs_mag1_bbox, rel_mag1_bbox])
        num_offsets = count_defined_values(
            [
                abs_mag1_offset,
                rel_mag1_offset,
                abs_current_mag_offset,
                rel_current_mag_offset,
            ]
        )
        num_sizes = count_defined_values([mag1_size, current_mag_size])
        if num_bboxes == 0:
            assert num_offsets != 0, "You must supply an offset or a bounding box."
            assert (
                num_sizes != 0
            ), "When supplying an offset, you must also supply a size. Alternatively, supply a bounding box."
            assert num_offsets == 1, "Only one offset can be supplied."
            assert num_sizes == 1, "Only one size can be supplied."
        else:
            assert num_bboxes == 1, "Only one bounding-box can be supplied."
            assert (
                num_offsets == 0
            ), "A bounding-box was supplied, you cannot also supply an offset."
            assert (
                num_sizes == 0
            ), "A bounding-box was supplied, you cannot also supply a size."

        if abs_mag1_bbox is not None:
            return abs_mag1_bbox

        if rel_mag1_bbox is not None:
            return rel_mag1_bbox.offset(self.bounding_box.topleft)

        else:
            mag_vec = self._mag.to_vec3_int()
            if rel_current_mag_offset is not None:
                abs_mag1_offset = (
                    self.bounding_box.topleft
                    + Vec3Int(rel_current_mag_offset) * mag_vec
                )
            if abs_current_mag_offset is not None:
                abs_mag1_offset = Vec3Int(abs_current_mag_offset) * mag_vec
            if rel_mag1_offset is not None:
                abs_mag1_offset = self.bounding_box.topleft + rel_mag1_offset

            if current_mag_size is not None:
                mag1_size = Vec3Int(current_mag_size) * mag_vec

            assert abs_mag1_offset is not None, "No offset was supplied."
            assert mag1_size is not None, "No size was supplied."

            assert (
                len(self.bounding_box) == 3
            ), "The delivered offset and size are only usable for 3D views."

            return self.bounding_box.with_topleft(abs_mag1_offset).with_size(mag1_size)

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
        """Write data to the view at a specified location.

        This method writes array data to the view's region. If no position parameters
        are provided, data is written to the view's bounding box. The write location
        can be specified using either offset parameters or bounding boxes.

        Args:
            data (np.ndarray): The data to write. For single-channel data, shape should
                be (x, y, y). For multi-channel data, shape should be (channels, x, y, z).
                Shape must match the target region size.
            offset (Optional[Vec3IntLike], optional): ⚠️ Deprecated. Use relative_offset or
                absolute_offset instead. Defaults to None.
            json_update_allowed (bool, optional): Whether to allow updating JSON metadata.
                Set to False when executing writes in parallel. Defaults to True.
            relative_offset (Optional[Vec3IntLike], optional): Offset relative to view's
                position in Mag(1) coordinates. Defaults to None.
            absolute_offset (Optional[Vec3IntLike], optional): Absolute offset in Mag(1)
                coordinates. Defaults to None.
            relative_bounding_box (Optional[NDBoundingBox], optional): Bounding box relative
                to view's position in Mag(1) coordinates. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox], optional): Absolute bounding box
                in Mag(1) coordinates. Defaults to None.

        Raises:
            AssertionError: If:
                - View is read-only
                - Data dimensions don't match the target region
                - Number of channels doesn't match the dataset
                - Write region is outside the view's bounding box
                - Multiple positioning parameters are provided

        Examples:
            ```python
            import numpy as np
            from webknossos.dataset import Dataset, View

            # Write to entire view's region
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            data = np.zeros(view.bounding_box.in_mag(view.mag).size)
            view.write(data)

            # Write with relative offset
            data = np.ones((50, 50, 5))  # Smaller region
            view.write(data, relative_offset=(10, 10, 0))

            # Write multi-channel data
            rgb_data = np.zeros((3, 100, 100, 10))  # 3 channels
            rgb_view = color_layer.get_mag("1").get_view(size=(100, 100, 10))
            rgb_view.write(rgb_data)

            # Write with absolute bounding box
            bbox = BoundingBox((0, 0, 0), (100, 100, 10))
            view.write(data, absolute_bounding_box=bbox)
            ```

        Note:
            - Only one positioning parameter (offset or bounding_box) should be used
            - All coordinates are in Mag(1) except for the deprecated offset
            - For compressed data, writes should align with compression blocks
            - For segmentation layers, call refresh_largest_segment_id() after writing
            - The view's magnification affects the actual data resolution
            - Data shape must match the target region size
        """
        assert not self.read_only, "Cannot write data to an read_only View"

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
            if len(data.shape) == len(self.bounding_box):
                shape_in_current_mag = data.shape
            else:
                shape_in_current_mag = data.shape[1:]

            absolute_bounding_box = (
                self.bounding_box.with_size(shape_in_current_mag)
                .from_mag_to_mag1(self._mag)
                .with_topleft(self.bounding_box.topleft)
            )

        if (absolute_bounding_box or relative_bounding_box) is not None:
            data_shape = None
        else:
            data_shape = Vec3Int(data.shape[-3:])

        if offset is not None:
            if self._mag == Mag(1):
                alternative = "Since this is a View in Mag(1), please use view.write(relative_offset=my_vec)"
            else:
                alternative = (
                    "Since this is a View, please use the coordinates in Mag(1) instead, e.g. "
                    + "view.write(relative_offset=my_vec * view.mag.to_vec3_int())"
                )

            warnings.warn(
                "[DEPRECATION] Using view.write(offset=my_vec) is deprecated. "
                + "Please use relative_offset or absolute_offset instead. "
                + alternative,
                DeprecationWarning,
            )

        num_channels = self._array.info.num_channels
        if len(data.shape) == len(self.bounding_box):
            assert (
                num_channels == 1
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data (1)"
        else:
            assert (
                num_channels == data.shape[0]
            ), f"The number of channels of the dataset ({num_channels}) does not match the number of channels of the passed data ({data.shape[0]})"

        mag1_bbox = self._get_mag1_bbox(
            rel_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            rel_mag1_bbox=relative_bounding_box,
            abs_mag1_bbox=absolute_bounding_box,
            current_mag_size=data_shape,
        )
        if json_update_allowed:
            assert self.bounding_box.contains_bbox(
                mag1_bbox
            ), f"The bounding box to write {mag1_bbox} is larger than the view's bounding box {self.bounding_box}"

        current_mag_bbox = mag1_bbox.in_mag(self._mag)

        if self._is_compressed():
            for current_mag_bbox, chunked_data in self._prepare_compressed_write(
                current_mag_bbox, data, json_update_allowed
            ):
                self._array.write(current_mag_bbox, chunked_data)
        else:
            self._array.write(current_mag_bbox, data)

    def _prepare_compressed_write(
        self,
        current_mag_bbox: NDBoundingBox,
        data: np.ndarray,
        json_update_allowed: bool = True,
    ) -> Iterator[Tuple[NDBoundingBox, np.ndarray]]:
        """This method takes an arbitrary sized chunk of data with an accompanying bbox,
        divides these into chunks of shard_shape size and delegates
        the preparation to _prepare_compressed_write_chunk."""

        chunked_bboxes = current_mag_bbox.chunk(
            self._array.info.shard_shape,
            chunk_border_alignments=self._array.info.shard_shape,
        )
        for chunked_bbox in chunked_bboxes:
            source_slice: Any
            if len(data.shape) == len(current_mag_bbox):
                source_slice = chunked_bbox.offset(
                    -current_mag_bbox.topleft
                ).to_slices()
            else:
                source_slice = (slice(None, None),) + chunked_bbox.offset(
                    -current_mag_bbox.topleft
                ).to_slices()

            yield self._prepare_compressed_write_chunk(
                chunked_bbox, data[source_slice], json_update_allowed
            )

    def _prepare_compressed_write_chunk(
        self,
        current_mag_bbox: NDBoundingBox,
        data: np.ndarray,
        json_update_allowed: bool = True,
    ) -> Tuple[NDBoundingBox, np.ndarray]:
        """This method takes an arbitrary sized chunk of data with an accompanying bbox
        (ideally not larger than a shard) and enlarges that chunk to fit the shard it
        resides in (by reading the entire shard data and writing the passed data ndarray
        into the specified volume). That way, the returned data can be written as a whole
        shard which is a requirement for compressed writes."""

        aligned_bbox = current_mag_bbox.align_with_mag(
            self._array.info.shard_shape, ceil=True
        )

        if current_mag_bbox != aligned_bbox:
            # The data bbox should either be aligned or match the dataset's bounding box:
            current_mag_view_bbox = self.bounding_box.in_mag(self._mag)
            if (
                json_update_allowed
                and current_mag_bbox
                != current_mag_view_bbox.intersected_with(aligned_bbox)
            ):
                warnings.warn(
                    _BLOCK_ALIGNMENT_WARNING,
                )

            aligned_data = self._read_without_checks(aligned_bbox)

            index_slice = (slice(None, None),) + current_mag_bbox.offset(
                -aligned_bbox.topleft
            ).to_slices()
            # overwrite the specified data
            aligned_data[index_slice] = data
            return aligned_bbox, aligned_data

        return current_mag_bbox, data

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
        """Read data from the view at a specified location.

        This method provides flexible ways to read data from the view's region. If no
        parameters are provided, it reads the entire view's bounding box. The region
        to read can be specified using either offset+size combinations or bounding boxes.

        Args:
            offset (Optional[Vec3IntLike], optional): ⚠️ Deprecated. Use relative_offset or
                absolute_offset instead. Defaults to None.
            size (Optional[Vec3IntLike], optional): Size of region to read. Specified in
                Mag(1) coordinates unless used with deprecated offset. Defaults to None.
            relative_offset (Optional[Vec3IntLike], optional): Offset relative to the view's
                position in Mag(1) coordinates. Must be used with size. Defaults to None.
            absolute_offset (Optional[Vec3IntLike], optional): Absolute offset in Mag(1)
                coordinates. Must be used with size. Defaults to None.
            relative_bounding_box (Optional[NDBoundingBox], optional): Bounding box relative
                to the view's position in Mag(1) coordinates. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox], optional): Absolute bounding box
                in Mag(1) coordinates. Defaults to None.

        Returns:
            np.ndarray: The requested data as a numpy array. The shape will be either
                (channels, x, y, z) for multi-channel data or (x, y, z) for
                single-channel data. Areas outside the dataset are zero-padded.

        Raises:
            AssertionError: If incompatible parameters are provided (e.g., both
                offset+size and bounding_box) or if the requested region is empty.

        Examples:
            ```python
            # Read entire view's data
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            data = view.read()  # Returns (x, y, z) array for single-channel data

            # Read with relative offset and size
            data = view.read(
                relative_offset=(10, 10, 0),  # Offset from view's position
                size=(50, 50, 10)            # Size in Mag(1) coordinates
            )

            # Read with absolute bounding box
            bbox = BoundingBox((0, 0, 0), (100, 100, 10))
            data = view.read(absolute_bounding_box=bbox)

            # Read from multi-channel data
            view = color_layer.get_mag("1").get_view(size=(100, 100, 10))
            data = view.read()  # Returns (channels, x, y, z) array
            ```

        Note:
            - Use only one method to specify the region (offset+size or bounding_box)
            - All coordinates are in Mag(1) except for the deprecated offset parameter
            - For multi-channel data, the returned array has shape (C, X, Y, Z)
            - For single-channel data, the returned array has shape (X, Y, Z)
            - Regions outside the dataset are automatically zero-padded
            - The view's magnification affects the actual data resolution
            - Data shape must match the target region size
        """
        current_mag_size: Optional[Vec3IntLike]
        mag1_size: Optional[Vec3IntLike]
        if absolute_bounding_box is None and relative_bounding_box is None:
            if offset is None:
                if size is None:
                    assert (
                        relative_offset is None and absolute_offset is None
                    ), "You must supply size, when reading with an offset."
                    absolute_bounding_box = self.bounding_box
                    current_mag_size = None
                    mag1_size = None
                else:
                    if relative_offset is None and absolute_offset is None:
                        if type(self) is View:
                            offset_param = "relative_offset"
                        else:
                            offset_param = "absolute_offset"
                        warnings.warn(
                            "[DEPRECATION] Using view.read(size=my_vec) only with a size is deprecated. "
                            + f"Please use view.read({offset_param}=(0, 0, 0), size=size_vec * view.mag.to_vec3_int()) instead.",
                            DeprecationWarning,
                        )
                        current_mag_size = size
                        mag1_size = None
                    else:
                        current_mag_size = None
                        mag1_size = size
            else:
                view_class = type(self).__name__
                if type(self) is View:
                    offset_param = "relative_offset"
                else:
                    offset_param = "absolute_offset"
                if self._mag == Mag(1):
                    alternative = f"Since this is a {view_class} in Mag(1), please use view.read({offset_param}=my_vec, size=size_vec)"
                else:
                    alternative = (
                        f"Since this is a {view_class}, please use the coordinates in Mag(1) instead, e.g. "
                        + f"view.read({offset_param}=my_vec * view.mag.to_vec3_int(),  size=size_vec * view.mag.to_vec3_int())"
                    )

                warnings.warn(
                    "[DEPRECATION] Using view.read(offset=my_vec) is deprecated. "
                    + "Please use relative_offset or absolute_offset instead. "
                    + alternative,
                    DeprecationWarning,
                )

                if size is None:
                    absolute_bounding_box = self.bounding_box.offset(
                        self._mag.to_vec3_int() * offset
                    )
                    offset = None
                    current_mag_size = None
                    mag1_size = None
                else:
                    # (deprecated) offset and size are given
                    current_mag_size = size
                    mag1_size = None

            if all(
                i is None
                for i in [
                    offset,
                    absolute_offset,
                    relative_offset,
                    absolute_bounding_box,
                ]
            ):
                relative_offset = Vec3Int.zeros()
        else:
            assert (
                size is None
            ), "Cannot supply a size when using bounding_box in view.read()"
            current_mag_size = None
            mag1_size = None

        mag1_bbox = self._get_mag1_bbox(
            rel_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            rel_mag1_bbox=relative_bounding_box,
            abs_mag1_bbox=absolute_bounding_box,
            current_mag_size=current_mag_size,
            mag1_size=mag1_size,
        )
        assert not mag1_bbox.is_empty(), (
            f"The size ({mag1_bbox.size} in mag1) contains a zero. "
            + "All dimensions must be strictly larger than '0'."
        )
        assert mag1_bbox.topleft.is_positive(), f"The offset ({mag1_bbox.topleft} in mag1) must not contain negative dimensions."

        return self._read_without_checks(mag1_bbox.in_mag(self._mag))

    def read_xyz(
        self,
        relative_bounding_box: Optional[NDBoundingBox] = None,
        absolute_bounding_box: Optional[NDBoundingBox] = None,
    ) -> np.ndarray:
        """Read n-dimensional data and convert it to 3D XYZ format.

        This method is designed for handling n-dimensional data (n > 3) and converting
        it to strictly 3D data ordered as (X, Y, Z). It is primarily used internally
        by operations that require 3D data like downsampling, upsampling, and compression.

        When provided with a BoundingBox where additional dimensions (beyond X, Y, Z)
        have a shape of 1, it returns an array containing only the 3D spatial data.
        This ensures compatibility with operations that expect purely 3-dimensional input.

        Args:
            relative_bounding_box (Optional[NDBoundingBox], optional): Bounding box relative
                to view's position in Mag(1) coordinates. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox], optional): Absolute bounding box
                in Mag(1) coordinates. Defaults to None.

        Returns:
            np.ndarray: The requested data as a numpy array with dimensions ordered as
                (channels, X, Y, Z) for multi-channel data or (X, Y, Z) for single-channel
                data. Areas outside the dataset are zero-padded.

        Examples:
            ```python
            # Read entire view's data in XYZ order
            view = layer.get_mag("1").get_view(size=(100, 100, 10))
            xyz_data = view.read_xyz()  # Returns (X, Y, Z) array

            # Read with relative bounding box
            bbox = NDBoundingBox((10, 10, 0), (50, 50, 10), axis=("x", "y", "z"), index=(1, 2, 3))
            xyz_data = view.read_xyz(relative_bounding_box=bbox)
            ```

        Note:
            - If no bounding box is provided, reads the entire view's region
            - Only one bounding box parameter should be specified
            - The returned array's axes are ordered differently from read()
            - All coordinates are in Mag(1)
        """
        mag1_bbox = self._get_mag1_bbox(
            rel_mag1_bbox=relative_bounding_box,
            abs_mag1_bbox=absolute_bounding_box,
        )
        if isinstance(mag1_bbox, BoundingBox):
            return self._read_without_checks(mag1_bbox.in_mag(self._mag))

        data = self._read_without_checks(mag1_bbox.in_mag(self._mag))
        # transform data to xyz order
        data = np.moveaxis(
            data,
            mag1_bbox.index_xyz,
            [1, 2, 3],
        )
        data = data.squeeze(axis=tuple(range(4, len(data.shape))))
        return data

    def read_bbox(self, bounding_box: Optional[BoundingBox] = None) -> np.ndarray:
        """⚠️ Deprecated. Please use `read()` with `relative_bounding_box` or `absolute_bounding_box` in Mag(1) instead.
        The user can specify the `bounding_box` in the current mag of the requested data.
        See `read()` for more details.
        """

        view_class = type(self).__name__
        if type(self) is View:
            offset_param = "relative_bounding_box"
        else:
            offset_param = "absolute_bounding_box"
        if self._mag == Mag(1):
            alternative = f"Since this is a {view_class} in Mag(1), please use view.read({offset_param}=bbox)"
        else:
            alternative = (
                f"Since this is a {view_class}, please use the bbox in Mag(1) instead, e.g. "
                + f"view.read({offset_param}=bbox.from_mag_to_mag1(view.mag))"
            )

        warnings.warn(
            "[DEPRECATION] read_bbox() (with a bbox in the current mag) is deprecated. "
            + "Please use read() with relative_bounding_box or absolute_bounding_box in Mag(1) instead. "
            + alternative,
            DeprecationWarning,
        )
        if bounding_box is None:
            return self.read()
        else:
            return self.read(bounding_box.topleft, bounding_box.size)

    def _read_without_checks(
        self,
        current_mag_bbox: NDBoundingBox,
    ) -> np.ndarray:
        data = self._array.read(current_mag_bbox)
        return data

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
    ) -> "View":
        """Create a new view restricted to a specified region.

        This method returns a new View instance that represents a subset of the current
        view's data. The new view can be specified using various coordinate systems
        and can optionally be made read-only.

        Args:
            offset (Optional[Vec3IntLike], optional): ⚠️ Deprecated. Use relative_offset or
                absolute_offset instead. Defaults to None.
            size (Optional[Vec3IntLike], optional): Size of the new view. Must be specified
                when using any offset parameter. Defaults to None.
            relative_offset (Optional[Vec3IntLike], optional): Offset relative to current
                view's position in Mag(1) coordinates. Must be used with size. Defaults to None.
            absolute_offset (Optional[Vec3IntLike], optional): Absolute offset in Mag(1)
                coordinates. Must be used with size. Defaults to None.
            relative_bounding_box (Optional[NDBoundingBox], optional): Bounding box relative
                to current view's position in Mag(1) coordinates. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox], optional): Absolute bounding box
                in Mag(1) coordinates. Defaults to None.
            read_only (Optional[bool], optional): Whether the new view should be read-only.
                If None, inherits from parent view. Defaults to None.

        Returns:
            View: A new View instance representing the specified region.

        Raises:
            AssertionError: If:
                - Multiple positioning parameters are provided
                - Size is missing when using offset parameters
                - Non-read-only subview requested from read-only parent
                - Non-read-only subview extends beyond parent's bounds

        Examples:
            ```python
            # Create view from MagView
            mag1 = layer.get_mag("1")
            view = mag1.get_view(absolute_offset=(10, 20, 30), size=(100, 200, 300))

            # Create subview within bounds
            sub_view = view.get_view(
                relative_offset=(50, 60, 70),
                size=(10, 120, 230)
            )

            # Create read-only subview (can extend beyond bounds)
            large_view = view.get_view(
                relative_offset=(50, 60, 70),
                size=(999, 120, 230),
                read_only=True
            )

            # Use bounding box instead of offset+size
            bbox = BoundingBox((10, 10, 0), (50, 50, 10))
            bbox_view = view.get_view(relative_bounding_box=bbox)
            ```

        Note:
            - Use only one method to specify the region (offset+size or bounding_box)
            - All coordinates are in Mag(1) except for the deprecated offset
            - Non-read-only views must stay within parent view's bounds
            - Read-only views can extend beyond parent view's bounds
            - The view's magnification affects the actual data resolution
        """
        if read_only is None:
            read_only = self.read_only
        else:
            assert (
                read_only or not self.read_only
            ), "Failed to get subview. The calling view is read_only. Therefore, the subview also has to be read_only."

        current_mag_size: Optional[Vec3IntLike]
        mag1_size: Optional[Vec3IntLike]
        if relative_bounding_box is None and absolute_bounding_box is None:
            if offset is None:
                if size is None:
                    assert (
                        relative_offset is None and absolute_offset is None
                    ), "You must supply a size, when using get_view with an offset."
                    current_mag_size = None
                    mag1_size = self.bounding_box.size
                else:
                    if relative_offset is None and absolute_offset is None:
                        if type(self) is View:
                            offset_param = "relative_offset"
                        else:
                            offset_param = "absolute_offset"
                        warnings.warn(
                            "[DEPRECATION] Using view.get_view(size=my_vec) only with a size is deprecated. "
                            + f"Please use view.get_view({offset_param}=(0, 0, 0), size=size_vec * view.mag.to_vec3_int()) instead.",
                            DeprecationWarning,
                        )
                        current_mag_size = size
                        mag1_size = None
                    else:
                        current_mag_size = None
                        mag1_size = size
            else:
                view_class = type(self).__name__
                if type(self) is View:
                    offset_param = "relative_offset"
                else:
                    offset_param = "absolute_offset"
                if self._mag == Mag(1):
                    alternative = f"Since this is a {view_class} in Mag(1), please use view.get_view({offset_param}=my_vec, size=size_vec)"
                else:
                    alternative = (
                        f"Since this is a {view_class}, please use the coordinates in Mag(1) instead, e.g. "
                        + f"view.get_view({offset_param}=my_vec * view.mag.to_vec3_int(),  size=size_vec * view.mag.to_vec3_int())"
                    )

                warnings.warn(
                    "[DEPRECATION] Using view.get_view(offset=my_vec) is deprecated. "
                    + "Please use relative_offset or absolute_offset instead. "
                    + alternative,
                    DeprecationWarning,
                )

                if size is None:
                    current_mag_size = None
                    mag1_size = self.bounding_box.size
                else:
                    # (deprecated) offset and size are given
                    current_mag_size = size
                    mag1_size = None

            if offset is None and relative_offset is None and absolute_offset is None:
                relative_offset = Vec3Int.zeros()
        else:
            assert (
                size is None
            ), "Cannot supply a size when using bounding_box in view.get_view()"
            current_mag_size = None
            mag1_size = None

        mag1_bbox = self._get_mag1_bbox(
            abs_mag1_bbox=absolute_bounding_box,
            rel_mag1_bbox=relative_bounding_box,
            rel_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            current_mag_size=current_mag_size,
            mag1_size=mag1_size,
        )
        if not self.bounding_box.is_empty():
            assert not mag1_bbox.is_empty(), (
                f"The size ({mag1_bbox.size} in mag1) contains a zero. "
                + "All dimensions must be strictly larger than '0'."
            )
        assert mag1_bbox.topleft.is_positive(), f"The offset ({mag1_bbox.topleft} in mag1) must not contain negative dimensions."

        if not read_only:
            assert self.bounding_box.contains_bbox(mag1_bbox), (
                f"The bounding box of the new subview {mag1_bbox} is larger than the view's bounding box {self.bounding_box}. "
                + "This is only allowed for read-only views."
            )

            current_mag_bbox = mag1_bbox.in_mag(self._mag)
            current_mag_aligned_bbox = current_mag_bbox.align_with_mag(
                self._array.info.shard_shape, ceil=True
            )
            # The data bbox should either be aligned or match the dataset's bounding box:
            current_mag_view_bbox = self.bounding_box.in_mag(self._mag)
            if current_mag_bbox != current_mag_view_bbox.intersected_with(
                current_mag_aligned_bbox, dont_assert=True
            ):
                warnings.warn(
                    "[WARNING] get_view() was called without block alignment. "
                    + "Please only use sequentially, parallel access across such views is error-prone.",
                    UserWarning,
                )

        return View(
            self._path,
            bounding_box=mag1_bbox,
            mag=self._mag,
            data_format=self._data_format,
            read_only=read_only,
        )

    def get_buffered_slice_writer(
        self,
        offset: Optional[Vec3IntLike] = None,
        buffer_size: int = 32,
        dimension: int = 2,  # z
        # json_update_allowed enables the update of the bounding box and rewriting of the properties json.
        # It should be False when parallel access is intended.
        json_update_allowed: bool = True,
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        use_logging: bool = False,
    ) -> "BufferedSliceWriter":
        """Get a buffered writer for efficiently writing data slices.

        Creates a BufferedSliceWriter that allows efficient writing of data slices by
        buffering multiple slices before performing the actual write operation.

         Args:
            offset (Optional[Vec3IntLike]): Starting position for writing in the dataset.
                Defaults to None.
            buffer_size (int): Number of slices to buffer before performing a write.
                Defaults to 32.
            dimension (int): Axis along which to write slices (0=x, 1=y, 2=z).
                Defaults to 2 (z-axis).
            json_update_allowed (bool): Whether to allow updating the bounding box and
                datasource-properties.json. Should be False for parallel access. Defaults to True.
            relative_offset (Optional[Vec3IntLike]): Offset in mag1 coordinates, relative
                to the current view's position. Mutually exclusive with absolute_offset.
                Defaults to None.
            absolute_offset (Optional[Vec3IntLike]): Offset in mag1 coordinates in
                absolute dataset coordinates. Mutually exclusive with relative_offset.
                Defaults to None.
            relative_bounding_box (Optional[NDBoundingBox]): Bounding box in mag1
                coordinates, relative to the current view's offset. Mutually exclusive
                with absolute_bounding_box. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox]): Bounding box in mag1
                coordinates in absolute dataset coordinates. Mutually exclusive with
                relative_bounding_box. Defaults to None.
            use_logging (bool): Whether to enable logging of write operations.
                Defaults to False.

        Returns:
            BufferedSliceWriter: A writer object for buffered slice writing.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))

            # Create a buffered writer with default settings
            with view.get_buffered_slice_writer() as writer:
                # Write slices efficiently
                for z in range(10):
                    slice_data = np.zeros((100, 100))  # Your slice data
                    writer.send(slice_data)

            # Create a writer with custom buffer size and offset
            with view.get_buffered_slice_writer(
                buffer_size=5,
                relative_offset=(10, 10, 0)
            )
            ```

        Note:
            - Larger buffer sizes can improve performance but use more memory
            - Remember to use the writer in a context manager
            - Only one positioning parameter should be specified
        """
        from ._utils.buffered_slice_writer import BufferedSliceWriter

        assert (
            not self._read_only
        ), "Cannot get a buffered slice writer on a read-only view."

        return BufferedSliceWriter(
            view=self,
            offset=offset,
            json_update_allowed=json_update_allowed,
            buffer_size=buffer_size,
            dimension=dimension,
            relative_offset=relative_offset,
            absolute_offset=absolute_offset,
            relative_bounding_box=relative_bounding_box,
            absolute_bounding_box=absolute_bounding_box,
            use_logging=use_logging,
        )

    def get_buffered_slice_reader(
        self,
        offset: Optional[Vec3IntLike] = None,
        size: Optional[Vec3IntLike] = None,
        buffer_size: int = 32,
        dimension: int = 2,  # z
        *,
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        use_logging: bool = False,
    ) -> "BufferedSliceReader":
        """Get a buffered reader for efficiently reading data slices.

        Creates a BufferedSliceReader that allows efficient reading of data slices by
        buffering multiple slices in memory. This is particularly useful when reading
        large datasets slice by slice.

        Args:
            offset (Optional[Vec3IntLike]): Starting position for reading in the dataset.
                Defaults to None.
            size (Optional[Vec3IntLike]): Size of the region to read in voxels.
                Defaults to None.
            buffer_size (int): Number of slices to buffer in memory at once.
                Defaults to 32.
            dimension (int): Axis along which to read slices (0=x, 1=y, 2=z).
                Defaults to 2 (z-axis).
            relative_bounding_box (Optional[NDBoundingBox]): Bounding box in mag1 coordinates,
                relative to the current view's offset. Mutually exclusive with
                absolute_bounding_box. Defaults to None.
            absolute_bounding_box (Optional[NDBoundingBox]): Bounding box in mag1 coordinates
                in absolute dataset coordinates. Mutually exclusive with
                relative_bounding_box. Defaults to None.
            use_logging (bool): Whether to enable logging of read operations.

        Returns:
            BufferedSliceReader: A reader object that yields data slices.

        Examples:
            ```python
            view = layer.get_mag("1").get_view(size=(100, 100, 10))

            # Create a reader with default settings (z-slices)
            with view.get_buffered_slice_reader() as reader:
                for slice_data in reader:
                    process_slice(slice_data)

            # Read y-slices with custom buffer size
            with view.get_buffered_slice_reader(
                buffer_size=10,
                dimension=1,  # y-axis
                relative_offset=(10, 0, 0)
            ) as reader:
            ```

        Note:
            - Larger buffer sizes improve performance but use more memory
            - Choose dimension based on your data access pattern
            - Only one positioning parameter should be specified
            - The reader can be used as an iterator
        """
        from ._utils.buffered_slice_reader import BufferedSliceReader

        return BufferedSliceReader(
            view=self,
            offset=offset,
            size=size,
            buffer_size=buffer_size,
            dimension=dimension,
            relative_bounding_box=relative_bounding_box,
            absolute_bounding_box=absolute_bounding_box,
            use_logging=use_logging,
        )

    def for_each_chunk(
        self,
        func_per_chunk: Callable[[Tuple["View", int]], None],
        chunk_shape: Optional[Vec3IntLike] = None,  # in Mag(1)
        executor: Optional[Executor] = None,
        progress_desc: Optional[str] = None,
        *,
        chunk_size: Optional[Vec3IntLike] = None,  # deprecated
    ) -> None:
        """Process each chunk of the view with a given function.

        Divides the view into chunks and applies a function to each chunk, optionally in parallel.
        This is useful for processing large datasets in manageable pieces, with optional
        progress tracking and parallel execution.

        Args:
            func_per_chunk (Callable[[Tuple[View, int]], None]): Function to apply to each chunk.
                Takes a tuple of (chunk_view, chunk_index) as argument. The chunk_index can be
                used for progress tracking or logging.
            chunk_shape (Optional[Vec3IntLike], optional): Size of each chunk in Mag(1) coordinates.
                If None, uses one chunk per file based on the dataset's file dimensions.
                Defaults to None.
            executor (Optional[Executor], optional): Executor for parallel processing.
                If None, processes chunks sequentially. Defaults to None.
            progress_desc (Optional[str], optional): Description for progress bar.
                If None, no progress bar is shown. Defaults to None.
            chunk_size (Optional[Vec3IntLike], optional): ⚠️ Deprecated. Use chunk_shape instead.
                Defaults to None.

        Examples:
            ```python
            from webknossos.utils import named_partial

            # Define processing function
            def process_chunk(args: Tuple[View, int], threshold: float) -> None:
                chunk_view, chunk_idx = args
                data = chunk_view.read()
                # Process data...
                chunk_view.write(processed_data)
                print(f"Processed chunk {chunk_idx}")

            # Sequential processing with progress bar
            view.for_each_chunk(
                named_partial(process_chunk, threshold=0.5),
                chunk_shape=(64, 64, 64),
                progress_desc="Processing chunks"
            )
            ```

        Note:
            - Each chunk is processed independently, making this suitable for parallel execution
            - For non-read-only views, chunks must align with file boundaries
            - Progress tracking shows total volume processed
            - Memory usage depends on chunk_shape and parallel execution settings
            - When using an executor, ensure thread/process safety in func_per_chunk
            - The view's magnification affects the actual data resolution
        """
        if chunk_shape is None:
            if chunk_size is not None:
                warn_deprecated("chunk_size", "chunk_shape")
                chunk_shape = Vec3Int(chunk_size)
                self._check_chunk_shape(chunk_shape, read_only=self.read_only)
            else:
                chunk_shape = self._get_file_dimensions_mag1()
        else:
            chunk_shape = Vec3Int(chunk_shape)
            self._check_chunk_shape(chunk_shape, read_only=self.read_only)

        job_args = []
        for i, chunk in enumerate(self.bounding_box.chunk(chunk_shape, chunk_shape)):
            chunk_view = self.get_view(
                absolute_offset=chunk.topleft,
                size=chunk.size,
            )
            job_args.append((chunk_view, i))

        # execute the work for each chunk
        if executor is None:
            if progress_desc is None:
                for args in job_args:
                    func_per_chunk(args)
            else:
                with get_rich_progress() as progress:
                    task = progress.add_task(
                        progress_desc, total=self.bounding_box.volume()
                    )
                    for args in job_args:
                        func_per_chunk(args)
                        current_view: View = args[0]
                        progress.update(
                            task, advance=current_view.bounding_box.volume()
                        )
        else:
            wait_and_ensure_success(
                executor.map_to_futures(func_per_chunk, job_args),
                executor=executor,
                progress_desc=progress_desc,
            )

    def map_chunk(
        self,
        func_per_chunk: Callable[["View"], Any],
        chunk_shape: Optional[Vec3IntLike] = None,  # in Mag(1)
        executor: Optional[Executor] = None,
        progress_desc: Optional[str] = None,
    ) -> List[Any]:
        """Process each chunk of the view and collect results.

        Similar to for_each_chunk(), but collects and returns the results from each chunk.
        Useful for parallel data analysis or feature extraction where results need to be
        aggregated.

        Args:
            func_per_chunk (Callable[[View], Any]): Function to apply to each chunk.
                Takes a chunk view as argument and returns a result of any type.
            chunk_shape (Optional[Vec3IntLike], optional): Size of each chunk in Mag(1) coordinates.
                If None, uses one chunk per file based on the dataset's file dimensions.
                Defaults to None.
            executor (Optional[Executor], optional): Executor for parallel processing.
                If None, processes chunks sequentially. Defaults to None.
            progress_desc (Optional[str], optional): Description for progress bar.
                If None, no progress bar is shown. Defaults to None.

        Returns:
            List[Any]: List of results from processing each chunk, in chunk order.

        Examples:
            ```python
            from webknossos.utils import named_partial

            # Calculate statistics per chunk
            def chunk_statistics(view: View, min_value: float) -> Dict[str, float]:
                data = view.read()
                return {
                    "mean": data[data > min_value].mean(),
                    "std": data[data > min_value].std(),
                    "volume": view.bounding_box.volume()
                }

            # Sequential processing
            stats = view.map_chunk(
                named_partial(chunk_statistics, min_value=0.1),
                chunk_shape=(128, 128, 128)
            )

            # Aggregate results
            total_volume = sum(s["volume"] for s in stats)
            mean_values = [s["mean"] for s in stats]
            ```

        Note:
            - Results are collected in memory, consider memory usage for large datasets
            - Each chunk is processed independently, suitable for parallel execution
            - For non-read-only views, chunks must align with file boundaries
            - When using an executor, ensure thread/process safety in func_per_chunk
            - Results maintain chunk order regardless of execution order
        """
        if chunk_shape is None:
            chunk_shape = self._get_file_dimensions_mag1()
        else:
            chunk_shape = Vec3Int(chunk_shape)
            self._check_chunk_shape(chunk_shape, read_only=self.read_only)

        job_args = []
        for chunk in self.bounding_box.chunk(chunk_shape, chunk_shape):
            chunk_view = self.get_view(
                absolute_offset=chunk.topleft,
                size=chunk.size,
            )
            job_args.append(chunk_view)

        # execute the work for each chunk
        with get_executor_for_args(None, executor) as executor:
            results = wait_and_ensure_success(
                executor.map_to_futures(func_per_chunk, job_args),
                executor=executor,
                progress_desc=progress_desc,
            )

        return results

    def chunk(
        self,
        chunk_shape: VecIntLike,
        chunk_border_alignments: Optional[VecIntLike] = None,
        read_only: bool = False,
    ) -> Generator["View", None, None]:
        """Generate a sequence of sub-views by chunking the current view.

        Divides the view into smaller, regularly-sized chunks that can be processed
        independently. This is useful for parallel processing or when working with
        large datasets that don't fit in memory.

        Args:
            chunk_shape (VecIntLike): Size of each chunk in Mag(1) coordinates.
            chunk_border_alignments (Optional[VecIntLike], optional): Alignment of chunk
                borders in Mag(1) coordinates. If None, aligns to (0, 0, 0).
                Defaults to None.
            read_only (bool, optional): Whether the generated chunks should be read-only.
                Defaults to False.

        Yields:
            View: Sub-views representing each chunk of the original view.

        Examples:
        ```python
        # let 'mag1' be a `MagView`
        chunks = mag1.chunk(chunk_shape=(100, 100, 100), chunk_border_alignments=(50, 50, 50))
        ```
        """

        for chunk in self.bounding_box.chunk(chunk_shape, chunk_border_alignments):
            yield self.get_view(absolute_bounding_box=chunk, read_only=read_only)

    def for_zipped_chunks(
        self,
        func_per_chunk: Callable[[Tuple["View", "View", int]], None],
        target_view: "View",
        source_chunk_shape: Optional[Vec3IntLike] = None,  # in Mag(1)
        target_chunk_shape: Optional[Vec3IntLike] = None,  # in Mag(1)
        executor: Optional[Executor] = None,
        progress_desc: Optional[str] = None,
        *,
        source_chunk_size: Optional[Vec3IntLike] = None,  # deprecated
        target_chunk_size: Optional[Vec3IntLike] = None,  # deprecated
    ) -> None:
        """Process paired chunks from source and target views simultaneously.

        Chunks both the source (self) and target views, then applies a function to each
        corresponding pair of chunks. This is particularly useful for operations that
        transform data between views of different magnifications, like downsampling.

        Args:
            func_per_chunk (Callable[[Tuple[View, View, int]], None]): Function to apply
                to each chunk pair. Takes (source_chunk, target_chunk, index) as arguments.
            target_view (View): The target view to write transformed data to.
            source_chunk_shape (Optional[Vec3IntLike], optional): Size of source chunks
                in Mag(1). If None, uses maximum of source and target file dimensions.
                Defaults to None.
            target_chunk_shape (Optional[Vec3IntLike], optional): Size of target chunks
                in Mag(1). If None, uses maximum of source and target file dimensions.
                Defaults to None.
            executor (Optional[Executor], optional): Executor for parallel processing.
                If None, processes chunks sequentially. Defaults to None.
            progress_desc (Optional[str], optional): Description for progress bar.
                If None, no progress bar is shown. Defaults to None.
            source_chunk_size (Optional[Vec3IntLike], optional): ⚠️ Deprecated.
                Use source_chunk_shape instead. Defaults to None.
            target_chunk_size (Optional[Vec3IntLike], optional): ⚠️ Deprecated.
                Use target_chunk_shape instead. Defaults to None.

        Examples:
            ```python
            # Downsample data from Mag(1) to Mag(2)
            def downsample_chunk(args: Tuple[View, View, int]) -> None:
                source_chunk, target_chunk, idx = args
                data = source_chunk.read()
                downsampled = downsample_data(data)  # Your downsampling function
                target_chunk.write(downsampled)
                print(f"Processed chunk pair {idx}")

            # Process with default chunk sizes
            mag1_view.for_zipped_chunks(
                downsample_chunk,
                mag2_view,
                progress_desc="Downsampling data"
            )

            ```

        Note:
            - Source/target view size ratios must match chunk size ratios
            - Target chunks must align with file boundaries to avoid concurrent writes
            - Both views are chunked with matching strides
            - Progress tracks total volume processed
            - Memory usage depends on chunk sizes and parallel execution
        """
        if source_chunk_shape is None and source_chunk_size is not None:
            warn_deprecated("source_chunk_size", "source_chunk_shape")
            source_chunk_shape = source_chunk_size

        if target_chunk_shape is None and target_chunk_size is not None:
            warn_deprecated("target_chunk_size", "target_chunk_shape")
            target_chunk_shape = target_chunk_size

        if source_chunk_shape is None or target_chunk_shape is None:
            assert (
                source_chunk_shape is None and target_chunk_shape is None
            ), "Either both source_chunk_shape and target_chunk_shape must be given or none."
            source_chunk_shape = self._get_file_dimensions_mag1().pairmax(
                target_view._get_file_dimensions_mag1()
            )
            target_chunk_shape = source_chunk_shape
        else:
            source_chunk_shape = Vec3Int(source_chunk_shape)
            target_chunk_shape = Vec3Int(target_chunk_shape)
            self._check_chunk_shape(source_chunk_shape, read_only=True)
            target_view._check_chunk_shape(
                target_chunk_shape, read_only=target_view.read_only
            )

        if self.bounding_box.is_empty() or target_view.bounding_box.is_empty():
            return

        assert np.array_equal(
            self.bounding_box.size_xyz.to_np()
            / target_view.bounding_box.size_xyz.to_np(),
            source_chunk_shape.to_np() / target_chunk_shape.to_np(),
        ), (
            "Calling 'for_zipped_chunks' failed because the ratio of the view sizes "
            + f"(source size = {self.bounding_box.size}, target size = {target_view.bounding_box.size}) "
            + "must be equal to the ratio of the chunk sizes "
            + f"(source_chunk_shape in Mag(1) = {source_chunk_shape}, target_chunk_shape in Mag(1) = {target_chunk_shape})"
        )

        source_views = self.chunk(
            source_chunk_shape, source_chunk_shape, read_only=True
        )
        target_views = target_view.chunk(target_chunk_shape, target_chunk_shape)

        job_args = (
            (source_view, target_view, i)
            for i, (source_view, target_view) in enumerate(
                zip(source_views, target_views)
            )
        )

        # execute the work for each pair of chunks
        if executor is None:
            if progress_desc is None:
                for args in job_args:
                    func_per_chunk(args)
            else:
                with get_rich_progress() as progress:
                    task = progress.add_task(
                        progress_desc, total=self.bounding_box.volume()
                    )
                    for args in job_args:
                        func_per_chunk(args)
                        progress.update(task, advance=args[0].bounding_box.volume())
        else:
            wait_and_ensure_success(
                executor.map_to_futures(func_per_chunk, job_args),
                executor=executor,
                progress_desc=progress_desc,
            )

    def content_is_equal(
        self,
        other: "View",
        args: Optional[Namespace] = None,  # deprecated
        executor: Optional[Executor] = None,
    ) -> bool:
        """Compare the content of this view with another view.

        Performs a chunk-by-chunk comparison of the data in both views. This is more
        memory efficient than reading entire views at once for large datasets.

        Args:
            other (View): The view to compare against.
            args Optional[Namespace], optional): ⚠️ Deprecated. Arguments to pass to executor.
            executor (Optional[Executor], optional): Executor for parallel comparison.
                If None, compares sequentially. Defaults to None.
            progress_desc (Optional[str], optional): Description for progress bar.
                If None, no progress bar is shown. Defaults to None.

        Returns:
            bool: True if the content of both views is identical, False otherwise.

        Examples:
            ```python
            # Compare views sequentially
            if view1.content_is_equal(view2):
                print("Views are identical")
            ```

        Note:
            - Comparison is done chunk by chunk to manage memory usage
            - Views must have the same shape and data type
            - Returns False immediately if shapes or types don't match
            - Progress tracks total volume compared
            - Parallel execution can speed up comparison of large views
        """
        if args is not None:
            warn_deprecated(
                "args argument",
                "executor (e.g. via webknossos.utils.get_executor_for_args(args))",
            )

        if self.bounding_box.size != other.bounding_box.size:
            return False
        with get_executor_for_args(args, executor) as executor:
            try:
                self.for_zipped_chunks(
                    _assert_check_equality,
                    other,
                    executor=executor,
                    progress_desc="Comparing contents",
                )
            except AssertionError:
                return False
        return True

    def _is_compressed(self) -> bool:
        return self._array.info.compression_mode

    def get_dtype(self) -> np.dtype:
        """
        Returns the dtype per channel of the data. For example `uint8`.
        """
        return self._array.info.voxel_type

    def __enter__(self) -> "View":
        warnings.warn(
            "[DEPRECATION] Entering a View to open it is deprecated. The internal dataset will be opened automatically.",
            DeprecationWarning,
        )
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        pass

    def __repr__(self) -> str:
        return f"View({repr(self._path)}, bounding_box={self.bounding_box})"

    def _check_chunk_shape(self, chunk_shape: Vec3Int, read_only: bool) -> None:
        assert chunk_shape.is_positive(
            strictly_positive=True
        ), f"The passed parameter 'chunk_shape' {chunk_shape} contains at least one 0. This is not allowed."

        divisor = self.mag.to_vec3_int() * self._array.info.chunk_shape
        if not read_only:
            divisor *= self._array.info.chunks_per_shard
        assert chunk_shape % divisor == Vec3Int.zeros(), (
            f"The chunk_shape {chunk_shape} must be a multiple of "
            + f"mag*chunk_shape{'*chunks_per_shard' if not read_only else ''} of the view, "
            + f"which is {divisor})."
        )

    def _get_file_dimensions(self) -> Vec3Int:
        return self._array.info.shard_shape

    def _get_file_dimensions_mag1(self) -> Vec3Int:
        return Vec3Int(self._get_file_dimensions() * self.mag.to_vec3_int())

    @property
    def _array(self) -> BaseArray:
        if self._cached_array is None:
            self._cached_array = BaseArray.get_class(self._data_format).open(
                self._path.resolve()
            )
        return self._cached_array

    @_array.deleter
    def _array(self) -> None:
        if self._cached_array is not None:
            self._cached_array.close()
            self._cached_array = None

    def __del__(self) -> None:
        if hasattr(self, "_cached_array"):
            del self._cached_array

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_array"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_array"] = None
        self.__dict__ = d


def _copy_job(args: Tuple[View, View, int]) -> None:
    (source_view, target_view, _) = args
    # Copy the data form one view to the other in a buffered fashion
    target_view.write(source_view.read())

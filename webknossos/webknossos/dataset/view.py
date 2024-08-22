import warnings
from argparse import Namespace
from pathlib import Path
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

from webknossos.geometry.vec_int import VecInt, VecIntLike

from ..geometry import BoundingBox, Mag, NDBoundingBox, Vec3Int, Vec3IntLike
from ..utils import (
    count_defined_values,
    get_executor_for_args,
    get_rich_progress,
    wait_and_ensure_success,
    warn_deprecated,
)
from ._array import ArrayInfo, BaseArray, WKWArray

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
    """
    A `View` is essentially a bounding box to a region of a specific `StorageBackend` that also provides functionality.
    Write-operations are restricted to the bounding box.
    `View`s are designed to be easily passed around as parameters.
    A `View`, in its most basic form, does not have a reference to its `StorageBackend`.
    """

    _path: Path
    _array_info: ArrayInfo
    _bounding_box: Optional[NDBoundingBox]
    _read_only: bool
    _cached_array: Optional[BaseArray]
    _mag: Mag

    def __init__(
        self,
        path_to_mag_view: Path,
        array_info: ArrayInfo,
        bounding_box: Optional[
            NDBoundingBox
        ],  # in mag 1, absolute coordinates, optional only for mag_view since it overwrites the bounding_box property
        mag: Mag,
        read_only: bool = False,
    ):
        """
        Do not use this constructor manually. Instead use `View.get_view()` (also available on a `MagView`) to get a `View`.
        """
        self._path = path_to_mag_view
        self._array_info = array_info
        self._bounding_box = bounding_box
        self._read_only = read_only
        self._cached_array = None
        self._mag = mag

    @property
    def info(self) -> ArrayInfo:
        return self._array_info

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
        assert self._bounding_box is not None
        return self._bounding_box

    @property
    def mag(self) -> Mag:
        return self._mag

    @property
    def read_only(self) -> bool:
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
        """
        The user can specify where the data should be written.
        The default is to write the data to the view's bounding box.
        Alternatively, one can supply one of the following keywords:

        * `relative_offset` in Mag(1) -> only usable for 3D datasets
        * `absolute_offset` in Mag(1) -> only usable for 3D datasets
        * `relative_bounding_box` in Mag(1)
        * `absolute_bounding_box` in Mag(1)

        ⚠️ The `offset` parameter is deprecated.
        This parameter used to be relative for `View` and absolute for `MagView`,
        and specified in the mag of the respective view.

        Writing data to a segmentation layer manually does not automatically update the largest_segment_id. To set
        the largest segment id properly run the `refresh_largest_segment_id` method on your layer or set the
        `largest_segment_id` property manually..

        Example:

        ```python
        ds = Dataset(DS_PATH, voxel_size=(1, 1, 1))

        segmentation_layer = cast(
            SegmentationLayer,
            ds.add_layer("segmentation", SEGMENTATION_CATEGORY),
        )
        mag = segmentation_layer.add_mag(Mag(1))

        mag.write(data=MY_NP_ARRAY)

        segmentation_layer.refresh_largest_segment_id()
        ```

        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions.
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
            if len(data.shape) == len(self.bounding_box) + 1:
                shape_in_current_mag = data.shape[1:]
            else:
                shape_in_current_mag = data.shape

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

        num_channels = self._array_info.num_channels
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
            self.info.shard_shape,
            chunk_border_alignments=self.info.shard_shape,
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

        aligned_bbox = current_mag_bbox.align_with_mag(self.info.shard_shape, ceil=True)

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
        """
        The user can specify which data should be read.
        The default is to read all data of the view's bounding box.
        Alternatively, one can supply one of the following keyword argument combinations:
        * `relative_offset` and `size`, both in Mag(1)
        * `absolute_offset` and `size`, both in Mag(1)
        * `relative_bounding_box` in Mag(1)
        * `absolute_bounding_box` in Mag(1)
        * ⚠️ deprecated: `offset` and `size`, both in the current Mag.
          `offset` used to be relative for `View` and absolute for `MagView`

        If the specified bounding box exceeds the data on disk, the rest is padded with `0`.

        Returns the specified data as a `np.array`.


        Example:
        ```python
        import numpy as np

        # ...
        # let mag1 be a MagView
        view = mag1.get_view(absolute_offset=(10, 20, 30), size=(100, 200, 300))

        assert np.array_equal(
            view.read(absolute_offset=(0, 0, 0), size=(100, 200, 300)),
            view.read(),
        )
        ```
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
            # offset is asserted anyways in _get_mag1_bbox
            current_mag_size = None
            mag1_size = None

        mag1_bbox = self._get_mag1_bbox(
            rel_current_mag_offset=offset,
            rel_mag1_offset=relative_offset,
            abs_mag1_offset=absolute_offset,
            current_mag_size=current_mag_size,
            mag1_size=mag1_size,
            abs_mag1_bbox=absolute_bounding_box,
            rel_mag1_bbox=relative_bounding_box,
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
        """
        The user can specify the bounding box in the dataset's coordinate system.
        The default is to read all data of the view's bounding box.
        Alternatively, one can supply one of the following keyword arguments:
        * `relative_bounding_box` in Mag(1)
        * `absolute_bounding_box` in Mag(1)

        Returns the specified data as a `np.array`.
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
        """
        ⚠️ Deprecated. Please use `read()` with `relative_bounding_box` or `absolute_bounding_box` in Mag(1) instead.
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
        """
        Returns a view that is limited to the specified bounding box.
        The new view may exceed the bounding box of the current view only if `read_only` is set to `True`.

        The default is to return the same view as the current bounding box,
        in case of a `MagView` that's the layer bounding box.
        One can supply one of the following keyword argument combinations:

        * `relative_bounding_box`in Mag(1)
        * `absolute_bounding_box` in Mag(1)
        * `relative_offset` and `size`, both in Mag(1)
        * `absolute_offset` and `size`, both in Mag(1)
        * ⚠️ deprecated: `offset` and `size`, both in the current Mag.
          `offset` used to be relative for `View` and absolute for `MagView`

        Example:
        ```python
        # ...
        # let mag1 be a MagView
        view = mag1.get_view(absolute_offset=(10, 20, 30), size=(100, 200, 300))

        # works because the specified sub-view is completely in the bounding box of the view
        sub_view = view.get_view(relative_offset=(50, 60, 70), size=(10, 120, 230))

        # fails because the specified sub-view is not completely in the bounding box of the view
        invalid_sub_view = view.get_view(relative_offset=(50, 60, 70), size=(999, 120, 230))

        # works because `read_only=True`
        valid_sub_view = view.get_view(relative_offset=(50, 60, 70), size=(999, 120, 230), read_only=True)
        ```
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
                self.info.shard_shape, ceil=True
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
            self.info,
            bounding_box=mag1_bbox,
            mag=self._mag,
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
        """
        The returned writer buffers multiple slices before they are written to disk.
        As soon as the buffer is full, the data gets written to disk.

        Arguments:

        * The user can specify where the writer should start:
            * `relative_offset` in Mag(1)
            * `absolute_offset` in Mag(1)
            * `relative_bounding_box` in Mag(1)
            * `absolute_bounding_box` in Mag(1)
            * ⚠️ deprecated: `offset` in the current Mag,
              used to be relative for `View` and absolute for `MagView`
        * `buffer_size`: amount of slices that get buffered
        * `dimension`: dimension along which the data is sliced
          (x: `0`, y: `1`, z: `2`; default is `2`)).

        The writer must be used as context manager using the `with` syntax (see example below),
        which results in a generator consuming np.ndarray-slices via `writer.send(slice)`.
        Exiting the context will automatically flush any remaining buffered data to disk.

        Usage:
        ```python
        data_cube = ...
        view = ...
        with view.get_buffered_slice_writer() as writer:
            for data_slice in data_cube:
                writer.send(data_slice)
        ```
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
        """
        The returned reader yields slices of data along a specified axis.
        Internally, it reads multiple slices from disk at once and buffers the data.

        Arguments:

        * The user can specify where the writer should start:
            * `relative_bounding_box` in Mag(1)
            * `absolute_bounding_box` in Mag(1)
            * ⚠️ deprecated: `offset` and `size` in the current Mag,
              `offset` used to be relative for `View` and absolute for `MagView`
        * `buffer_size`: amount of slices that get buffered
        * `dimension`: dimension along which the data is sliced
          (x: `0`, y: `1`, z: `2`; default is `2`)).

        The reader must be used as a context manager using the `with` syntax (see example below).
        Entering the context returns an iterator yielding slices (np.ndarray).

        Usage:
        ```python
        view = ...
        with view.get_buffered_slice_reader() as reader:
            for slice_data in reader:
                ...
        ```
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
        """
        The view is chunked into multiple sub-views of size `chunk_shape` (in Mag(1)),
        by default one chunk per file.
        Then, `func_per_chunk` is performed on each sub-view.
        Besides the view, the counter `i` is passed to the `func_per_chunk`,
        which can be used for logging.
        Additional parameters for `func_per_chunk` can be specified using `functools.partial`.
        The computation of each chunk has to be independent of each other.
        Therefore, the work can be parallelized with `executor`.

        If the `View` is of type `MagView` only the bounding box from the properties is chunked.

        Example:
        ```python
        from webknossos.utils import named_partial

        def some_work(args: Tuple[View, int], some_parameter: int) -> None:
            view_of_single_chunk, i = args
            # perform operations on the view
            ...

        # ...
        # let 'mag1' be a `MagView`
        func = named_partial(some_work, some_parameter=42)
        mag1.for_each_chunk(
            func,
        )
        ```
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
        """
        The view is chunked into multiple sub-views of size `chunk_shape` (in Mag(1)),
        by default one chunk per file.
        Then, `func_per_chunk` is performed on each sub-view and the results are collected
        in a list.
        Additional parameters for `func_per_chunk` can be specified using `functools.partial`.
        The computation of each chunk has to be independent of each other.
        Therefore, the work can be parallelized with `executor`.

        If the `View` is of type `MagView` only the bounding box from the properties is chunked.

        Example:
        ```python
        from webknossos.utils import named_partial

        def some_work(view: View, some_parameter: int) -> None:
            # perform operations on the view
            ...

        # ...
        # let 'mag1' be a `MagView`
        func = named_partial(some_work, some_parameter=42)
        results = mag1.map_chunk(
            func,
        )
        ```
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
        """
        This method chunks the view into multiple sub-views of size `chunk_shape` (in Mag(1)).
        The `chunk_border_alignments` parameter specifies the alignment of the chunks.
        The default is to align the chunks to the origin (0, 0, 0).

        Example:
        ```python
        # ...
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
        """
        This method is similar to `for_each_chunk` in the sense that it delegates work to smaller chunks,
        given by `source_chunk_shape` and `target_chunk_shape` (both in Mag(1),
        by default using the larger of the source_views and the target_views file-sizes).
        However, this method also takes another view as a parameter. Both views are chunked simultaneously
        and a matching pair of chunks is then passed to the function that shall be executed.
        This is useful if data from one view should be (transformed and) written to a different view,
        assuming that the transformation of the data can be handled on chunk-level.
        Additionally to the two views, the counter `i` is passed to the `func_per_chunk`, which can be used for logging.

        The mapping of chunks from the source view to the target is bijective.
        The ratio between the size of the `source_view` (`self`) and the `source_chunk_shape` must be equal to
        the ratio between the `target_view` and the `target_chunk_shape`. This guarantees that the number of chunks
        in the `source_view` is equal to the number of chunks in the `target_view`.
        The `target_chunk_shape` must be a multiple of the file size on disk to avoid concurrent writes.

        Example use case: *downsampling from Mag(1) to Mag(2)*
        - size of the views: `16384³` (`8192³` in Mag(2) for `target_view`)
        - automatic chunk sizes: `2048³`, assuming  default file-lengths
          (`1024³` in Mag(2), which fits the default file-length of 32*32)
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
        return self.info.compression_mode

    def get_dtype(self) -> np.dtype:
        """
        Returns the dtype per channel of the data. For example `uint8`.
        """
        return self.info.voxel_type

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
        return f"View({repr(str(self._path))}, bounding_box={self.bounding_box})"

    def _check_chunk_shape(self, chunk_shape: Vec3Int, read_only: bool) -> None:
        assert chunk_shape.is_positive(
            strictly_positive=True
        ), f"The passed parameter 'chunk_shape' {chunk_shape} contains at least one 0. This is not allowed."

        divisor = self.mag.to_vec3_int() * self.info.chunk_shape
        if not read_only:
            divisor *= self.info.chunks_per_shard
        assert chunk_shape % divisor == Vec3Int.zeros(), (
            f"The chunk_shape {chunk_shape} must be a multiple of "
            + f"mag*chunk_shape{'*chunks_per_shard' if not read_only else ''} of the view, "
            + f"which is {divisor})."
        )

    def _get_file_dimensions(self) -> Vec3Int:
        return self.info.shard_shape

    def _get_file_dimensions_mag1(self) -> Vec3Int:
        return Vec3Int(self._get_file_dimensions() * self.mag.to_vec3_int())

    @property
    def _array(self) -> BaseArray:
        if self._cached_array is None:
            cls_array = BaseArray.get_class(self.info.data_format)
            self._cached_array = cls_array(self._path)
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

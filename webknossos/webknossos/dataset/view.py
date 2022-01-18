import math
import warnings
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    Union,
)

import cluster_tools
import numpy as np
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from wkw import Dataset, wkw

from webknossos.geometry import BoundingBox, Mag, Vec3Int, Vec3IntLike
from webknossos.utils import get_rich_progress, wait_and_ensure_success

if TYPE_CHECKING:
    from webknossos.dataset._utils.buffered_slice_reader import BufferedSliceReader
    from webknossos.dataset._utils.buffered_slice_writer import BufferedSliceWriter


class View:
    """
    A `View` is essentially a bounding box to a region of a specific `wkw.Dataset` that also provides functionality.
    Write-operations are restricted to the bounding box.
    `View`s are designed to be easily passed around as parameters.
    A `View`, in its most basic form, does not have a reference to its `Dataset`.
    """

    def __init__(
        self,
        path_to_mag_view: Path,
        header: wkw.Header,
        bounding_box: Optional[
            BoundingBox
        ],  # in mag 1, absolute coordinates, optional only for mag_view since it overwrites the bounding_box property
        mag: Mag,
        read_only: bool = False,
    ):
        """
        Do not use this constructor manually. Instead use `View.get_view()` (also available on a `MagView`) to get a `View`.
        """
        self._path = path_to_mag_view
        self._header: wkw.Header = header
        self._bounding_box = bounding_box
        self._read_only = read_only
        self._cached_wkw_dataset = None
        self._mag = mag

    @property
    def header(self) -> wkw.Header:
        return self._header

    @property
    def bounding_box(self) -> BoundingBox:
        assert self._bounding_box is not None
        return self._bounding_box

    @property
    def mag(self) -> Mag:
        return self._mag

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def global_offset(self) -> Vec3Int:
        """⚠️ Deprecated, use `view.bounding_box.in_mag(view.mag).topleft` instead."""
        warnings.warn(
            "[DEPRECATION] view.global_offset is deprecated. "
            + "Since this is a View, please use "
            + "view.bounding_box.in_mag(view.mag).topleft instead."
        )
        return self.bounding_box.in_mag(self._mag).topleft

    @property
    def size(self) -> Vec3Int:
        """⚠️ Deprecated, use `view.bounding_box.in_mag(view.mag).size` instead."""
        warnings.warn(
            "[DEPRECATION] view.size is deprecated. "
            + "Since this is a View, please use "
            + "view.bounding_box.in_mag(view.mag).size instead."
        )
        return self.bounding_box.in_mag(self._mag).size

    def _get_mag1_bbox(
        self,
        abs_mag1_bbox: Optional[BoundingBox] = None,
        rel_mag1_bbox: Optional[BoundingBox] = None,
        abs_mag1_offset: Optional[Vec3IntLike] = None,
        rel_mag1_offset: Optional[Vec3IntLike] = None,
        mag1_size: Optional[Vec3IntLike] = None,
        abs_current_mag_offset: Optional[Vec3IntLike] = None,
        rel_current_mag_offset: Optional[Vec3IntLike] = None,
        current_mag_size: Optional[Vec3IntLike] = None,
    ) -> BoundingBox:
        num_bboxes = _count_defined_values([abs_mag1_bbox, rel_mag1_bbox])
        num_offsets = _count_defined_values(
            [
                abs_mag1_offset,
                rel_mag1_offset,
                abs_current_mag_offset,
                rel_current_mag_offset,
            ]
        )
        num_sizes = _count_defined_values([mag1_size, current_mag_size])
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

        elif rel_mag1_bbox is not None:
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
            return BoundingBox(Vec3Int(abs_mag1_offset), Vec3Int(mag1_size))

    def write(
        self,
        data: np.ndarray,
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative, in current mag
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
    ) -> None:
        """
        Writes the `data` at the specified `relative_offset` or `absolute_offset`, both specified in Mag(1).

        ⚠️ The `offset` parameter is deprecated.
        This parameter used to be relative for `View` and absolute for `MagView`,
        and specified in the mag of the respective view.

        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions.
        """
        assert not self.read_only, "Cannot write data to an read_only View"

        if all(i is None for i in [offset, absolute_offset, relative_offset]):
            relative_offset = Vec3Int.zeros()

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
                + alternative
            )

        num_channels = self._header.num_channels
        if len(data.shape) == 3:
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
            current_mag_size=Vec3Int(data.shape[-3:]),
        )
        assert self.bounding_box.contains_bbox(
            mag1_bbox
        ), f"The bounding box to write {mag1_bbox} is larger than the view's bounding box {self.bounding_box}"

        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0]  # remove channel dimension for single-channel data

        current_mag_bbox = mag1_bbox.in_mag(self._mag)

        if self._is_compressed():
            current_mag_bbox, data = self._handle_compressed_write(
                current_mag_bbox, data
            )

        self._wkw_dataset.write(current_mag_bbox.topleft, data)

    def _handle_compressed_write(
        self, current_mag_bbox: BoundingBox, data: np.ndarray
    ) -> Tuple[BoundingBox, np.ndarray]:
        aligned_bbox = current_mag_bbox.align_with_mag(
            Mag(self.header.file_len * self.header.block_len), ceil=True
        )

        if current_mag_bbox != aligned_bbox:

            # The data bbox should either be aligned or match the dataset's bounding box:
            current_mag_view_bbox = self.bounding_box.in_mag(self._mag)
            if current_mag_bbox != current_mag_view_bbox.intersected_with(aligned_bbox):
                warnings.warn(
                    "Warning: write() was called on a compressed mag without block alignment. "
                    + "Performance will be degraded as the data has to be padded first.",
                    RuntimeWarning,
                )

            aligned_data = self._read_without_checks(aligned_bbox)

            index_slice = (slice(None, None),) + current_mag_bbox.offset(
                -aligned_bbox.topleft
            ).to_slices()
            # overwrite the specified data
            aligned_data[index_slice] = data
            return aligned_bbox, aligned_data
        else:
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
        relative_bounding_box: Optional[BoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[BoundingBox] = None,  # in mag1
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
        view = mag1.get_view(absolute_offset(10, 20, 30), size=(100, 200, 300))

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
                    current_mag_size = None
                    mag1_size = self.bounding_box.size
                else:
                    if relative_offset is None and absolute_offset is None:
                        if type(self) == View:
                            offset_param = "relative_offset"
                        else:
                            offset_param = "absolute_offset"
                        warnings.warn(
                            "[DEPRECATION] Using view.read(size=my_vec) only with a size is deprecated. "
                            + f"Please use view.read({offset_param}=(0, 0, 0), size=size_vec * view.mag.to_vec3_int()) instead."
                        )
                        current_mag_size = size
                        mag1_size = None
                    else:
                        current_mag_size = None
                        mag1_size = size
            else:
                view_class = type(self).__name__
                if type(self) == View:
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
                    + alternative
                )

                if size is None:
                    current_mag_size = None
                    mag1_size = self.bounding_box.size
                else:
                    # (deprecated) offset and size are given
                    current_mag_size = size
                    mag1_size = None

            if all(i is None for i in [offset, absolute_offset, relative_offset]):
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

        return self._read_without_checks(mag1_bbox.in_mag(self._mag))

    def read_bbox(self, bounding_box: Optional[BoundingBox] = None) -> np.ndarray:
        """
        ⚠️ Deprecated. Please use `read()` with `relative_bounding_box` or `absolute_bounding_box` in Mag(1) instead.
        The user can specify the `bounding_box` in the current mag of the requested data.
        See `read()` for more details.
        """

        view_class = type(self).__name__
        if type(self) == View:
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
            + alternative
        )
        if bounding_box is None:
            return self.read()
        else:
            return self.read(bounding_box.topleft, bounding_box.size)

    def _read_without_checks(
        self,
        current_mag_bbox: BoundingBox,
    ) -> np.ndarray:
        data = self._wkw_dataset.read(
            current_mag_bbox.topleft.to_np(), current_mag_bbox.size.to_np()
        )
        return data

    def get_view(
        self,
        offset: Optional[Vec3IntLike] = None,
        size: Optional[Vec3IntLike] = None,
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        read_only: Optional[bool] = None,
    ) -> "View":
        """
        Returns a view that is limited to the specified bounding box.
        The new view may exceed the bounding box of the current view only if `read_only` is set to `True`.

        The default is to return the same view as the current bounding box,
        in case of a `MagView` that's the layer bounding box.
        One can supply one of the following keyword argument combinations:
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

        if offset is None:
            if size is None:
                assert (
                    relative_offset is None and absolute_offset is None
                ), "You must supply a size, when using get_view with an offset."
                current_mag_size = None
                mag1_size = self.bounding_box.size
            else:
                if relative_offset is None and absolute_offset is None:
                    if type(self) == View:
                        offset_param = "relative_offset"
                    else:
                        offset_param = "absolute_offset"
                    warnings.warn(
                        "[DEPRECATION] Using view.get_view(size=my_vec) only with a size is deprecated. "
                        + f"Please use view.get_view({offset_param}=(0, 0, 0), size=size_vec * view.mag.to_vec3_int()) instead."
                    )
                    current_mag_size = size
                    mag1_size = None
                else:
                    current_mag_size = None
                    mag1_size = size
        else:
            view_class = type(self).__name__
            if type(self) == View:
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
                + alternative
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

        mag1_bbox = self._get_mag1_bbox(
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

        if not read_only:
            assert self.bounding_box.contains_bbox(mag1_bbox), (
                f"The bounding box of the new subview {mag1_bbox} is larger than the view's bounding box {self.bounding_box}. "
                + "This is only allowed for read-only views."
            )

            current_mag_bbox = mag1_bbox.in_mag(self._mag)
            current_mag_aligned_bbox = current_mag_bbox.align_with_mag(
                Mag(self.header.file_len * self.header.block_len), ceil=True
            )
            # The data bbox should either be aligned or match the dataset's bounding box:
            current_mag_view_bbox = self.bounding_box.in_mag(self._mag)
            if current_mag_bbox != current_mag_view_bbox.intersected_with(
                current_mag_aligned_bbox, dont_assert=True
            ):
                warnings.warn(
                    "Warning: get_view() was called without block alignment. "
                    + "Please only use sequentially, parallel access across such views is error-prone.",
                    RuntimeWarning,
                )

        return View(
            self._path,
            self.header,
            bounding_box=mag1_bbox,
            mag=self._mag,
            read_only=read_only,
        )

    def get_buffered_slice_writer(
        self,
        offset: Vec3IntLike = Vec3Int(0, 0, 0),
        buffer_size: int = 32,
        dimension: int = 2,  # z
    ) -> "BufferedSliceWriter":
        """
        The BufferedSliceWriter buffers multiple slices before they are written to disk.
        The amount of slices that get buffered is specified by `buffer_size`.
        As soon as the buffer is full, the data gets written to disk.

        The user can specify along which dimension the data is sliced by using the parameter `dimension`.
        To slice along the x-axis use `0`, for the y-axis use `1`, or for the z-axis use `2` (default: dimension=2).

        The BufferedSliceWriter must be used as context manager using the `with` syntax (see example below),
        which results in a generator consuming np.ndarray-slices via `writer.send(slice)`.
        Exiting the context will automatically flush any remaining buffered data to disk.

        Usage:
        data_cube = ...
        view = ...
        with view.get_buffered_slice_writer() as writer:
            for data_slice in data_cube:
                writer.send(data_slice)

        """
        from webknossos.dataset._utils.buffered_slice_writer import BufferedSliceWriter

        assert (
            not self._read_only
        ), "Cannot get a buffered slice writer on a read-only view."

        return BufferedSliceWriter(
            view=self,
            offset=Vec3Int(offset),
            buffer_size=buffer_size,
            dimension=dimension,
        )

    def get_buffered_slice_reader(
        self,
        offset: Vec3IntLike = Vec3Int(0, 0, 0),
        size: Optional[Vec3IntLike] = None,
        buffer_size: int = 32,
        dimension: int = 2,  # z
    ) -> "BufferedSliceReader":
        """
        The BufferedSliceReader yields slices of data along a specified axis.
        Internally, it reads multiple slices from disk at once and buffers the data.
        The amount of slices that get buffered is specified by `buffer_size`.

        The user can specify along which dimension the data is sliced by using the parameter `dimension`.
        To slice along the x-axis use `0`, for the y-axis use `1`, or for the z-axis use `2` (default: dimension=2).

        The BufferedSliceReader must be used as a context manager using the `with` syntax (see example below).
        Entering the context returns a generator with yields slices (np.ndarray).

        Usage:
        view = ...
        with view.get_buffered_slice_reader() as reader:
            for slice_data in reader:
                ...

        """
        from webknossos.dataset._utils.buffered_slice_reader import BufferedSliceReader

        return BufferedSliceReader(
            view=self,
            offset=Vec3Int(offset),
            size=Vec3Int(size)
            if size is not None
            else self.bounding_box.in_mag(self._mag).size,
            buffer_size=buffer_size,
            dimension=dimension,
        )

    def for_each_chunk(
        self,
        work_on_chunk: Callable[[Tuple["View", int]], None],
        chunk_size: Vec3IntLike,  # in current mag
        executor: Optional[
            Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor]
        ] = None,
        progress_desc: Optional[str] = None,
    ) -> None:
        """
        The view is chunked into multiple sub-views of size `chunk_size`.
        Then, `work_on_chunk` is performed on each sub-view.
        Besides the view, the counter 'i' is passed to the 'work_on_chunk',
        which can be used for logging. Additional parameter for 'work_on_chunk' can be specified.
        The computation of each chunk has to be independent of each other.
        Therefore, the work can be parallelized with `executor`.

        If the `View` is of type `MagView`, only the bounding box from the properties is chunked.

        Example:
        ```python
        from webknossos.utils import get_executor_for_args, named_partial

        def some_work(args: Tuple[View, int], some_parameter: int) -> None:
            view_of_single_chunk, i = args
            # perform operations on the view
            ...

        # ...
        # let 'mag1' be a `MagView`
        view = mag1.get_view()
        func = named_partial(some_work, some_parameter=42)
        view.for_each_chunk(
            func,
            chunk_size=(100, 100, 100),  # Use mag1._get_file_dimensions() if the size of the chunks should match the size of the files on disk
        )
        ```
        """

        chunk_size = Vec3Int(chunk_size)
        _check_chunk_size(chunk_size)
        mag1_chunk_size = chunk_size * self.mag.to_vec3_int()

        job_args = []
        for i, chunk in enumerate(
            self.bounding_box.chunk(mag1_chunk_size, mag1_chunk_size)
        ):
            chunk_view = self.get_view(
                absolute_offset=chunk.topleft,
                size=chunk.size,
            )
            job_args.append((chunk_view, i))

        # execute the work for each chunk
        if executor is None:
            if progress_desc is None:
                for args in job_args:
                    work_on_chunk(args)
            else:
                with get_rich_progress() as progress:
                    task = progress.add_task(
                        progress_desc, total=self.bounding_box.volume()
                    )
                    for args in job_args:
                        work_on_chunk(args)
                        current_view: View = args[0]
                        progress.update(
                            task, advance=current_view.bounding_box.volume()
                        )
        else:
            wait_and_ensure_success(
                executor.map_to_futures(work_on_chunk, job_args), progress_desc
            )

    def for_zipped_chunks(
        self,
        work_on_chunk: Callable[[Tuple["View", "View", int]], None],
        target_view: "View",
        source_chunk_size: Vec3IntLike,  # in current mag
        target_chunk_size: Vec3IntLike,  # in target view mag
        executor: Optional[
            Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor]
        ] = None,
        progress_desc: Optional[str] = None,
    ) -> None:
        """
        This method is similar to 'for_each_chunk' in the sense, that it delegates work to smaller chunks.
        However, this method also takes another view as a parameter. Both views are chunked simultaneously
        and a matching pair of chunks is then passed to the function that shall be executed.
        This is useful if data from one view should be (transformed and) written to a different view,
        assuming that the transformation of the data can be handled on chunk-level.
        Additionally to the two views, the counter 'i' is passed to the 'work_on_chunk',
        which can be used for logging.
        The mapping of chunks from the source view to the target is bijective.
        The ratio between the size of the source_view (self) and the source_chunk_size must be equal to
        the ratio between the target_view and the target_chunk_size. This guarantees that the number of chunks
        in the source_view is equal to the number of chunks in the target_view.

        Example use case: downsampling:
        - size of source_view (Mag 1): (16384, 16384, 16384)
        - size of target_view (Mag 2): (8192, 8192, 8192)
        - source_chunk_size: (2048, 2048, 2048)
        - target_chunk_size: (1024, 1024, 1024) // this must be a multiple of the file size on disk to avoid concurrent writes
        """
        source_chunk_size = Vec3Int(source_chunk_size)
        target_chunk_size = Vec3Int(target_chunk_size)

        _check_chunk_size(source_chunk_size)
        _check_chunk_size(target_chunk_size)

        mag1_source_chunk_size = source_chunk_size * self.mag.to_vec3_int()
        mag1_target_chunk_size = target_chunk_size * target_view.mag.to_vec3_int()

        assert (
            not self.bounding_box.is_empty()
        ), "Calling 'for_zipped_chunks' failed because the size of the source view contains a 0."
        assert (
            not target_view.bounding_box.is_empty()
        ), "Calling 'for_zipped_chunks' failed because the size of the target view contains a 0."
        assert np.array_equal(
            self.bounding_box.size.to_np() / target_view.bounding_box.size.to_np(),
            mag1_source_chunk_size.to_np() / mag1_target_chunk_size.to_np(),
        ), (
            "Calling 'for_zipped_chunks' failed because the ratio of the view sizes "
            + f"(source size = {self.bounding_box.size}, target size = {target_view.bounding_box.size}) "
            + "must be equal to the ratio of the chunk sizes "
            + f"(source_chunk_size in Mag(1) = {mag1_source_chunk_size}, target_chunk_size in Mag(1) = {mag1_target_chunk_size})"
        )

        assert not any(
            target_chunk_size.to_np()
            % (target_view.header.file_len * target_view.header.block_len)
        ), f"Calling for_zipped_chunks failed. The target_chunk_size ({target_chunk_size}) must be a multiple of file_len*block_len of the target view ({target_view.header.file_len * target_view.header.block_len})"

        job_args = []
        source_chunks = self.bounding_box.chunk(
            mag1_source_chunk_size, mag1_source_chunk_size
        )
        target_chunks = target_view.bounding_box.chunk(
            mag1_target_chunk_size, mag1_target_chunk_size
        )

        for i, (source_chunk, target_chunk) in enumerate(
            zip(source_chunks, target_chunks)
        ):
            source_chunk_view = self.get_view(
                absolute_offset=source_chunk.topleft,
                size=source_chunk.size,
                read_only=True,
            )
            target_chunk_view = target_view.get_view(
                absolute_offset=target_chunk.topleft,
                size=target_chunk.size,
            )

            job_args.append((source_chunk_view, target_chunk_view, i))

        # execute the work for each pair of chunks
        if executor is None:
            if progress_desc is None:
                for args in job_args:
                    work_on_chunk(args)
            else:
                with get_rich_progress() as progress:
                    task = progress.add_task(
                        progress_desc, total=self.bounding_box.volume()
                    )
                    for args in job_args:
                        work_on_chunk(args)
                        progress.update(task, advance=args[0].bounding_box.volume())
        else:
            wait_and_ensure_success(
                executor.map_to_futures(work_on_chunk, job_args), progress_desc
            )

    def _is_compressed(self) -> bool:
        return (
            self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4
            or self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4HC
        )

    def get_dtype(self) -> type:
        """
        Returns the dtype per channel of the data. For example `uint8`.
        """
        return self.header.voxel_type

    def __enter__(self) -> "View":
        warnings.warn(
            "[DEPRECATION] Entering a View to open it is deprecated. The internal dataset will be opened automatically."
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
        return repr(f"View({self._path}, bounding_box={self.bounding_box})")

    @property
    def _wkw_dataset(self) -> wkw.Dataset:
        if self._cached_wkw_dataset is None:
            self._cached_wkw_dataset = Dataset.open(
                str(self._path)
            )  # No need to pass the header to the wkw.Dataset
        return self._cached_wkw_dataset

    @_wkw_dataset.deleter
    def _wkw_dataset(self) -> None:
        if self._cached_wkw_dataset is not None:
            self._cached_wkw_dataset.close()
            self._cached_wkw_dataset = None

    def __del__(self) -> None:
        del self._cached_wkw_dataset

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_cached_wkw_dataset"]
        return d

    def __setstate__(self, d: Dict[str, Any]) -> None:
        d["_cached_wkw_dataset"] = None
        self.__dict__ = d


def _check_chunk_size(chunk_size: Vec3Int) -> None:
    assert chunk_size is not None

    if 0 in chunk_size:
        raise AssertionError(
            f"The passed parameter 'chunk_size' {chunk_size} contains at least one 0. This is not allowed."
        )
    if not np.all(
        np.array([math.log2(size).is_integer() for size in np.array(chunk_size)])
    ):
        raise AssertionError(
            f"Each element of the passed parameter 'chunk_size' {chunk_size} must be a power of 2.."
        )
    if (np.array(chunk_size) % (32, 32, 32)).any():
        raise AssertionError(
            f"The passed parameter 'chunk_size' {chunk_size} must be a multiple of (32, 32, 32)."
        )


def _count_defined_values(values: Iterable[Optional[Any]]) -> int:
    return sum(i is not None for i in values)

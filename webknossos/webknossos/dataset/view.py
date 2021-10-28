import math
import warnings
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

import cluster_tools
import numpy as np
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from wkw import Dataset, wkw

from webknossos.geometry import BoundingBox, Vec3Int, Vec3IntLike
from webknossos.utils import wait_and_ensure_success

if TYPE_CHECKING:
    from webknossos.dataset._utils.buffered_slice_reader import BufferedSliceReader
    from webknossos.dataset._utils.buffered_slice_writer import BufferedSliceWriter


class View:
    """
    A `View` is essentially a bounding box to a region of a specific `wkw.Dataset` that also provides functionality.
    Read- and write-operations are restricted to the bounding box.
    `View`s are designed to be easily passed around as parameters.
    A `View`, in its most basic form, does not have a reference to the `webknossos.dataset.dataset.Dataset`.
    """

    def __init__(
        self,
        path_to_mag_view: Path,
        header: wkw.Header,
        size: Vec3IntLike,
        global_offset: Vec3IntLike,
        is_bounded: bool = True,
        read_only: bool = False,
        mag_view_bbox_at_creation: Optional[BoundingBox] = None,
    ):
        """
        Do not use this constructor manually. Instead use `webknossos.dataset.mag_view.MagView.get_view()` to get a `View`.
        """
        self._path = path_to_mag_view
        self._header: wkw.Header = header
        self._size: Vec3Int = Vec3Int(size)
        self._global_offset: Vec3Int = Vec3Int(global_offset)
        self._is_bounded = is_bounded
        self._read_only = read_only
        self._cached_wkw_dataset = None
        # The bounding box of the view is used to prevent warnings when writing compressed but unaligned data
        # directly at the borders of the bounding box.
        # A View is unable to get this information from the Dataset because it is detached from it.
        # Adding the bounding box as parameter is a workaround for this.
        # However, keep in mind that this bounding box is just a snapshot.
        # This bounding box is not updated if the bounding box of the dataset is updated.
        # Even though an outdated bounding box can lead to missed (or unwanted) warnings,
        # this is sufficient for our use case because such scenarios are unlikely and not critical.
        # This should not be misused to infer the size of the dataset because this might lead to problems.
        self._mag_view_bbox_at_creation = mag_view_bbox_at_creation

    @property
    def header(self) -> wkw.Header:
        return self._header

    @property
    def size(self) -> Vec3Int:
        return self._size

    @property
    def global_offset(self) -> Vec3Int:
        return self._global_offset

    @property
    def read_only(self) -> bool:
        return self._read_only

    def write(
        self,
        data: np.ndarray,
        offset: Vec3IntLike = Vec3Int(0, 0, 0),
    ) -> None:
        """
        Writes the `data` at the specified `offset` to disk.
        The `offset` is relative to `global_offset`.

        Note that writing compressed data which is not aligned with the blocks on disk may result in
        diminished performance, as full blocks will automatically be read to pad the write actions.
        """
        assert not self.read_only, "Cannot write data to an read_only View"

        offset = Vec3Int(offset)

        # assert the size of the parameter data is not in conflict with the attribute self.size
        data_dims = Vec3Int(data.shape[-3:])
        _assert_positive_dimensions(offset, data_dims)
        self._assert_bounds(offset, data_dims)

        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0]  # remove channel dimension

        # calculate the absolute offset
        absolute_offset = self.global_offset + offset

        if self._is_compressed():
            absolute_offset, data = self._handle_compressed_write(absolute_offset, data)

        self._wkw_dataset.write(absolute_offset.to_np(), data)

    def read(
        self,
        offset: Vec3IntLike = Vec3Int(0, 0, 0),
        size: Optional[Vec3IntLike] = None,
    ) -> np.ndarray:
        """
        The user can specify the `offset` and the `size` of the requested data.
        The `offset` is relative to `global_offset`.
        If no `size` is specified, the size of the view is used.
        If the specified bounding box exceeds the data on disk, the rest is padded with `0`.

        Returns the specified data as a `np.array`.


        Example:
        ```python
        import numpy as np

        # ...
        # let 'mag1' be a `MagView`
        view = mag1.get_view(offset(10, 20, 30), size=(100, 200, 300))

        assert np.array_equal(
            view.read(offset=(0, 0, 0), size=(100, 200, 300)),
            view.read(),
        )

        # works because the specified data is completely in the bounding box of the view
        some_data = view.read(offset=(50, 60, 70), size=(10, 120, 230))

        # fails because the specified data is not completely in the bounding box of the view
        more_data = view.read(offset=(50, 60, 70), size=(999, 120, 230))
        ```
        """

        offset = Vec3Int(offset)
        size = self.size if size is None else Vec3Int(size)

        # assert the parameter size is not in conflict with the attribute self.size
        _assert_positive_dimensions(offset, size)
        self._assert_bounds(offset, size)

        # calculate the absolute offset
        absolute_offset = self.global_offset + offset

        return self._read_without_checks(absolute_offset, size)

    def read_bbox(self, bounding_box: Optional[BoundingBox] = None) -> np.ndarray:
        """
        The user can specify the `bounding_box` of the requested data.
        See `read()` for more details.
        """
        if bounding_box is None:
            return self.read()
        else:
            return self.read(bounding_box.topleft, bounding_box.size)

    def _read_without_checks(
        self,
        absolute_offset: Vec3Int,
        size: Vec3Int,
    ) -> np.ndarray:
        data = self._wkw_dataset.read(absolute_offset.to_np(), size.to_np())
        return data

    def get_view(
        self,
        offset: Vec3IntLike = Vec3Int(0, 0, 0),
        size: Optional[Vec3IntLike] = None,
        read_only: Optional[bool] = None,
    ) -> "View":
        """
        Returns a view that is limited to the specified bounding box.
        The `offset` is relative to `global_offset`.
        If no `size` is specified, the size of the view is used.

        The `offset` and `size` may only exceed the bounding box of the current view, if `read_only` is set to `True`.


        If `read_only` is `True`, write operations are not allowed for the returned sub-view.

        Example:
        ```python
        # ...
        # let 'mag1' be a `MagView`
        view = mag1.get_view(offset(10, 20, 30), size=(100, 200, 300))

        # works because the specified sub-view is completely in the bounding box of the view
        sub_view = view.get_view(offset=(50, 60, 70), size=(10, 120, 230))

        # fails because the specified sub-view is not completely in the bounding box of the view
        invalid_sub_view = view.get_view(offset=(50, 60, 70), size=(999, 120, 230))

        # works because `read_only=True`
        invalid_sub_view = view.get_view(offset=(50, 60, 70), size=(999, 120, 230), read_only=True)
        ```
        """
        if read_only is None:
            read_only = self.read_only
        assert (
            read_only or read_only == self.read_only
        ), "Failed to get subview. The calling view is read_only. Therefore, the subview also has to be read_only."

        offset = Vec3Int(offset)
        size = self.size if size is None else Vec3Int(size)

        _assert_positive_dimensions(offset, size)
        self._assert_bounds(offset, size, not read_only)
        view_offset = self.global_offset + offset
        return View(
            self._path,
            self.header,
            size=size,
            global_offset=view_offset,
            is_bounded=True,
            read_only=read_only,
            mag_view_bbox_at_creation=self._mag_view_bounding_box_at_creation,
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
            size=Vec3Int(size) if size is not None else self.size,
            buffer_size=buffer_size,
            dimension=dimension,
        )

    def _assert_bounds(
        self,
        offset: Vec3Int,
        size: Vec3Int,
        strict: Optional[bool] = None,
    ) -> None:
        if strict is None:
            strict = self._is_bounded
        if strict and not BoundingBox((0, 0, 0), self.size).contains_bbox(
            BoundingBox(offset, size)
        ):
            raise AssertionError(
                f"Accessing data out of bounds: The passed parameter 'size' {size} exceeds the size of the current view ({self.size})"
            )

    def for_each_chunk(
        self,
        work_on_chunk: Callable[[Tuple["View", int]], None],
        chunk_size: Vec3IntLike,
        executor: Optional[
            Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor]
        ] = None,
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
        # This "view" object assures that the operation cannot exceed the bounding box of the properties.
        # `View.get_view()` returns a `View` of the same size as the current object (because of the default parameters).
        # `MagView.get_view()` returns a `View` with the bounding box from the properties.
        view = self.get_view()

        job_args = []

        for i, chunk in enumerate(
            BoundingBox(view.global_offset, view.size).chunk(chunk_size, chunk_size)
        ):
            relative_offset = chunk.topleft - view.global_offset
            chunk_view = view.get_view(
                offset=relative_offset,
                size=chunk.size,
            )
            job_args.append((chunk_view, i))

        # execute the work for each chunk
        if executor is None:
            for args in job_args:
                work_on_chunk(args)
        else:
            wait_and_ensure_success(executor.map_to_futures(work_on_chunk, job_args))

    def for_zipped_chunks(
        self,
        work_on_chunk: Callable[[Tuple["View", "View", int]], None],
        target_view: "View",
        source_chunk_size: Vec3IntLike,
        target_chunk_size: Vec3IntLike,
        executor: Optional[
            Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor]
        ] = None,
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

        source_view = self.get_view()
        target_view = target_view.get_view()

        source_offset = source_view.global_offset
        target_offset = target_view.global_offset
        source_chunk_size_np = source_chunk_size.to_np()
        target_chunk_size_np = target_chunk_size.to_np()

        assert not source_view.size.contains(
            0
        ), "Calling 'for_zipped_chunks' failed because the size of the source view contains a 0."
        assert not target_view.size.contains(
            0
        ), "Calling 'for_zipped_chunks' failed because the size of the target view contains a 0."
        assert np.array_equal(
            source_view.size.to_np() / target_view.size.to_np(),
            source_chunk_size_np / target_chunk_size_np,
        ), f"Calling 'for_zipped_chunks' failed because the ratio of the view sizes (source size = {source_view.size}, target size = {target_view.size}) must be equal to the ratio of the chunk sizes (source_chunk_size = {source_chunk_size}, source_chunk_size = {target_chunk_size}))"

        assert not any(
            target_chunk_size_np
            % (target_view.header.file_len * target_view.header.block_len)
        ), f"Calling for_zipped_chunks failed. The target_chunk_size ({target_chunk_size}) must be a multiple of file_len*block_len of the target view ({target_view.header.file_len * target_view.header.block_len})"

        job_args = []
        source_chunks = BoundingBox(source_offset, source_view.size).chunk(
            source_chunk_size, source_chunk_size
        )
        target_chunks = BoundingBox(target_offset, target_view.size).chunk(
            target_chunk_size, target_chunk_size
        )

        for i, (source_chunk, target_chunk) in enumerate(
            zip(source_chunks, target_chunks)
        ):
            # source chunk
            relative_source_offset = source_chunk.topleft - source_offset
            source_chunk_view = source_view.get_view(
                offset=relative_source_offset,
                size=source_chunk.size,
                read_only=True,
            )
            # target chunk
            relative_target_offset = target_chunk.topleft - target_offset
            target_chunk_view = target_view.get_view(
                size=target_chunk.size,
                offset=relative_target_offset,
            )

            job_args.append((source_chunk_view, target_chunk_view, i))

        # execute the work for each pair of chunks
        if executor is None:
            for args in job_args:
                work_on_chunk(args)
        else:
            wait_and_ensure_success(executor.map_to_futures(work_on_chunk, job_args))

    def _is_compressed(self) -> bool:
        return (
            self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4
            or self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4HC
        )

    def _handle_compressed_write(
        self, absolute_offset: Vec3Int, data: np.ndarray
    ) -> Tuple[Vec3Int, np.ndarray]:
        # calculate aligned bounding box
        file_bb = np.full(3, self.header.file_len * self.header.block_len)
        absolute_offset_np = np.array(absolute_offset)
        margin_to_top_left = absolute_offset_np % file_bb
        aligned_offset = absolute_offset_np - margin_to_top_left
        bottom_right = absolute_offset_np + np.array(data.shape[-3:])
        margin_to_bottom_right = (file_bb - (bottom_right % file_bb)) % file_bb
        is_bottom_right_aligned = bottom_right + margin_to_bottom_right
        aligned_shape = is_bottom_right_aligned - aligned_offset

        if (
            tuple(aligned_offset) != absolute_offset
            or tuple(aligned_shape) != data.shape[-3:]
        ):
            mag_view_bbox_at_creation = self._mag_view_bounding_box_at_creation

            # Calculate in which dimensions the data is aligned and in which dimensions it matches the bbox of the mag.
            is_top_left_aligned = aligned_offset == np.array(absolute_offset)
            is_bottom_right_aligned = is_bottom_right_aligned == bottom_right
            is_at_border_top_left = np.array(
                mag_view_bbox_at_creation.topleft
            ) == np.array(absolute_offset)
            is_at_border_bottom_right = (
                np.array(mag_view_bbox_at_creation.bottomright) == bottom_right
            )

            if not (
                np.all(np.logical_or(is_top_left_aligned, is_at_border_top_left))
                and np.all(
                    np.logical_or(is_bottom_right_aligned, is_at_border_bottom_right)
                )
            ):
                # the data is not aligned
                # read the aligned bounding box

                # We want to read the data at the absolute offset.
                # The absolute offset might be outside of the current view.
                # That is the case if the data is compressed but the view does not include the whole file on disk.
                # In this case we avoid checking the bounds because the aligned_offset and aligned_shape are calculated internally.
                warnings.warn(
                    "Warning: write() was called on a compressed mag without block alignment. Performance will be degraded as the data has to be padded first.",
                    RuntimeWarning,
                )
            aligned_data = self._read_without_checks(
                Vec3Int(aligned_offset), Vec3Int(aligned_shape)
            )

            index_slice = (
                slice(None, None),
                *(
                    slice(start, end)
                    for start, end in zip(
                        margin_to_top_left, bottom_right - aligned_offset
                    )
                ),
            )
            # overwrite the specified data
            aligned_data[tuple(index_slice)] = data
            return Vec3Int(aligned_offset), aligned_data
        else:
            return absolute_offset, data

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
        return repr(
            "View(%s, global_offset=%s, size=%s)"
            % (self._path, self.global_offset, self.size)
        )

    @property
    def _mag_view_bounding_box_at_creation(self) -> BoundingBox:
        assert self._mag_view_bbox_at_creation is not None
        return self._mag_view_bbox_at_creation

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


def _assert_positive_dimensions(offset: Vec3Int, size: Vec3Int) -> None:
    if any(x < 0 for x in offset):
        raise AssertionError(
            f"The offset ({offset}) contains a negative value. All dimensions must be larger or equal to '0'."
        )
    if any(x <= 0 for x in size):
        raise AssertionError(
            f"The size ({size}) contains a negative value (or zeros). All dimensions must be strictly larger than '0'."
        )


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

import math
from pathlib import Path
from types import TracebackType
from typing import Tuple, Optional, Type, Callable, Union, cast

import cluster_tools
import numpy as np
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from wkw import Dataset, wkw

from wkcuber.api.bounding_box import BoundingBox
from wkcuber.utils import wait_and_ensure_success


class View:
    """
    A `View` is essentially a bounding box to a section of a specific `wkw.Dataset` that also provides functionality.
    Read- and write-operations are restricted to the bounding box.
    `View`s are designed to be easily passed around as parameters.
    A `View`, in its most basic form, does not have a reference to the `wkcuber.api.dataset.WKDataset`.
    """

    def __init__(
        self,
        path_to_mag_dataset: Path,
        header: wkw.Header,
        size: Tuple[int, int, int],
        global_offset: Tuple[int, int, int] = (0, 0, 0),
        is_bounded: bool = True,
        read_only: bool = False,
    ):
        """
        Initializes a `View`. If `create` is `True`, a `wkw.Dataset` is created (see [webknossos-wrap (wkw)](https://github.com/scalableminds/webknossos-wrap)).

        Most of the time the `View` is used in the context of the dataset API, even though it could also be used without it.

        In the context of the dataset API, `View`s appear for example in the form of `wkcuber.api.mag_dataset.WKMagDataset` (which are also `View`s).
        Sub-views can be created with `View.get_view()`.
        """
        self.dataset: Optional[Dataset] = None
        self.path = path_to_mag_dataset
        self.header: wkw.Header = header
        self.size = size
        self.global_offset: Tuple[int, int, int] = global_offset
        self._is_bounded = is_bounded
        self.read_only = read_only
        self._is_opened = False

    def open(self) -> "View":
        """
        Opens the actual handles to the data on disk.
        A `MagDataset` has to be opened before it can be read or written to. However, the user does not
        have to open it explicitly because the API automatically opens it when it is needed.
        The user can choose to open it explicitly to avoid that handles are opened and closed automatically
        each time data is read or written.
        """
        if self._is_opened:
            raise Exception("Cannot open view: the view is already opened")
        else:
            self.dataset = Dataset.open(
                str(self.path)
            )  # No need to pass the header to the wkw.Dataset
            self._is_opened = True
        return self

    def close(self) -> None:
        """
        Complementary to `open`, this closes the handles to the data.

        See `open` for more information.
        """
        if not self._is_opened:
            raise Exception("Cannot close View: the view is not opened")
        else:
            assert self.dataset is not None  # because the View was opened
            self.dataset.close()
            self.dataset = None
            self._is_opened = False

    def write(
        self,
        data: np.ndarray,
        relative_offset: Tuple[int, int, int] = (0, 0, 0),
        allow_compressed_write: bool = False,
    ) -> None:
        """
        Writes the `data` at the specified `offset` to disk.
        The `offset` is relative to `global_offset`.

        If the data on disk is compressed, the passed `data` either has to be aligned with the files on disk
        or `allow_compressed_write` has to be `True`. If `allow_compressed_write` is `True`, `data` is padded by
        first reading the necessary padding from disk.
        In this particular case, reading data from outside the bounding box is allowed.
        """
        assert not self.read_only, "Cannot write data to an read_only View"

        was_opened = self._is_opened
        # assert the size of the parameter data is not in conflict with the attribute self.size
        self._assert_bounds(relative_offset, data.shape[-3:])

        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0]  # remove channel dimension

        # calculate the absolute offset
        absolute_offset = cast(
            Tuple[int, int, int],
            tuple(sum(x) for x in zip(self.global_offset, relative_offset)),
        )

        if self._is_compressed() and allow_compressed_write:
            absolute_offset, data = self._handle_compressed_write(absolute_offset, data)

        if not was_opened:
            self.open()
        assert self.dataset is not None  # because the View was opened

        self.dataset.write(absolute_offset, data)

        if not was_opened:
            self.close()

    def read(
        self,
        offset: Tuple[int, int, int] = (0, 0, 0),
        size: Tuple[int, int, int] = None,
    ) -> np.array:
        """
        The user can specify the `offset` and the `size` of the requested data.
        The `offset` is relative to `global_offset`.
        If no `size` is specified, the size of the view is used.
        If the specified bounding box exceeds the data on disk, the rest is padded with `0`.

        Retruns the specified data as a `np.array`.


        Example:
        ```python
        import numpy as np

        # ...
        # let 'mag1' be a `WKMagDataset`
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

        size = self.size if size is None else size

        # assert the parameter size is not in conflict with the attribute self.size
        self._assert_bounds(offset, size)

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        return self._read_without_checks(
            cast(Tuple[int, int, int], absolute_offset), size
        )

    def _read_without_checks(
        self,
        absolute_offset: Tuple[int, int, int],
        size: Tuple[int, int, int],
    ) -> np.array:
        was_opened = self._is_opened
        if not was_opened:
            self.open()
        assert self.dataset is not None  # because the View was opened

        data = self.dataset.read(absolute_offset, size)

        if not was_opened:
            self.close()

        return data

    def get_view(
        self,
        offset: Tuple[int, int, int] = None,
        size: Tuple[int, int, int] = None,
        read_only: bool = None,
    ) -> "View":
        """
        Returns a view that is limited to the specified bounding box.
        The `offset` is relative to `global_offset`.
        If no `size` is specified, the size of the view is used.

        If `read_only` is `True`, write operations are not allowed for the returned sub-view.

        Example:
        ```python
        # ...
        # let 'mag1' be a `WKMagDataset`
        view = mag1.get_view(offset(10, 20, 30), size=(100, 200, 300))

        # works because the specified sub-view is completely in the bounding box of the view
        sub_view = view.get_view(offset=(50, 60, 70), size=(10, 120, 230))

        # fails because the specified sub-view is not completely in the bounding box of the view
        invalid_sub_view = view.get_view(offset=(50, 60, 70), size=(999, 120, 230))
        ```
        """
        if read_only is None:
            read_only = self.read_only
        assert (
            read_only or read_only == self.read_only
        ), "Failed to get subview. The calling view is read_only. Therefore, the subview also has to be read_only."

        if offset is None:
            offset = (0, 0, 0)

        if size is None:
            size = self.size

        self._assert_bounds(offset, size)
        view_offset = cast(
            Tuple[int, int, int], tuple(self.global_offset + np.array(offset))
        )
        return View(
            self.path,
            self.header,
            size=size,
            global_offset=view_offset,
            is_bounded=True,
            read_only=read_only,
        )

    def _check_bounds(
        self, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> bool:
        for s1, s2, off in zip(self.size, size, offset):
            if s2 + off > s1 and self._is_bounded:
                return False
        if any(x < 0 for x in offset):
            return False
        if any(x < 0 for x in size):
            return False
        return True

    def _assert_bounds(
        self, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> None:
        if not self._check_bounds(offset, size):
            raise AssertionError(
                f"Accessing data out of bounds: The passed parameter 'size' {size} exceeds the size of the current view ({self.size})"
            )

    def for_each_chunk(
        self,
        work_on_chunk: Callable[[Tuple["View", int]], None],
        chunk_size: Tuple[int, int, int],
        executor: Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor],
    ) -> None:
        """
        The view is chunked into multiple sub-views of size `chunk_size`.
        Then, `work_on_chunk` is performed on each sub-view.
        Besides the view, the counter 'i' is passed to the 'work_on_chunk',
        which can be used for logging. additional parameter for 'work_on_chunk' can be specified.
        The computation of each chunk has to be independent of each other.
        Therefore, the work can be parallelized with `executor`.

        Example:
        ```python
        from wkcuber.utils import get_executor_for_args, named_partial

        def some_work(args: Tuple[View, int], some_parameter: int) -> None:
            view_of_single_chunk, i = args
            # perform operations on the view
            ...

        # ...
        # let 'mag1' be a `WKMagDataset`
        view = mag1.get_view()
        with get_executor_for_args(None) as executor:
            func = named_partial(advanced_chunk_job, some_parameter=42)
            view.for_each_chunk(
                func,
                chunk_size=(100, 100, 100),  # Use mag1._get_file_dimensions() if the size of the chunks should match the size of the files on disk
                executor=executor,
            )
        ```
        """
        self._check_chunk_size(chunk_size)

        job_args = []

        for i, chunk in enumerate(
            BoundingBox(self.global_offset, self.size).chunk(
                chunk_size, list(chunk_size)
            )
        ):
            relative_offset = cast(
                Tuple[int, int, int],
                tuple(np.array(chunk.topleft) - np.array(self.global_offset)),
            )
            view = self.get_view(
                offset=relative_offset,
                size=cast(Tuple[int, int, int], tuple(chunk.size)),
            )
            view._is_bounded = True
            job_args.append((view, i))

        # execute the work for each chunk
        wait_and_ensure_success(executor.map_to_futures(work_on_chunk, job_args))

    def for_zipped_chunks(
        self,
        work_on_chunk: Callable[[Tuple["View", "View", int]], None],
        target_view: "View",
        source_chunk_size: Tuple[int, int, int],
        target_chunk_size: Tuple[int, int, int],
        executor: Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor],
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
        source_offset = np.array(self.global_offset)
        target_offset = np.array(target_view.global_offset)
        source_chunk_size_np = np.array(source_chunk_size)
        target_chunk_size_np = np.array(target_chunk_size)

        assert np.all(
            np.array(self.size)
        ), "Calling 'for_zipped_chunks' failed because the size of the source view contains a 0."
        assert np.all(
            np.array(target_view.size)
        ), "Calling 'for_zipped_chunks' failed because the size of the target view contains a 0."
        assert np.array_equal(
            np.array(self.size) / np.array(target_view.size),
            source_chunk_size_np / target_chunk_size_np,
        ), f"Calling 'for_zipped_chunks' failed because the ratio of the view sizes (source size = {self.size}, target size = {target_view.size}) must be equal to the ratio of the chunk sizes (source_chunk_size = {source_chunk_size}, source_chunk_size = {target_chunk_size}))"

        assert not any(
            target_chunk_size_np
            % (target_view.header.file_len * target_view.header.block_len)
        ), f"Calling for_zipped_chunks failed. The target_chunk_size ({target_chunk_size}) must be a multiple of file_len*block_len of the target view ({target_view.header.file_len * target_view.header.block_len})"

        job_args = []
        source_chunks = BoundingBox(source_offset, self.size).chunk(
            source_chunk_size_np, list(source_chunk_size_np)
        )
        target_chunks = BoundingBox(target_offset, target_view.size).chunk(
            target_chunk_size, list(target_chunk_size)
        )

        for i, (source_chunk, target_chunk) in enumerate(
            zip(source_chunks, target_chunks)
        ):
            # source chunk
            relative_source_offset = np.array(source_chunk.topleft) - source_offset
            source_chunk_view = self.get_view(
                offset=cast(Tuple[int, int, int], tuple(relative_source_offset)),
                size=cast(Tuple[int, int, int], tuple(source_chunk.size)),
                read_only=True,
            )
            # target chunk
            relative_target_offset = np.array(target_chunk.topleft) - target_offset
            target_chunk_view = target_view.get_view(
                size=cast(Tuple[int, int, int], tuple(target_chunk.size)),
                offset=cast(Tuple[int, int, int], tuple(relative_target_offset)),
            )

            job_args.append((source_chunk_view, target_chunk_view, i))

        # execute the work for each pair of chunks
        wait_and_ensure_success(executor.map_to_futures(work_on_chunk, job_args))

    def _check_chunk_size(self, chunk_size: Tuple[int, int, int]) -> None:
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

    def _is_compressed(self) -> bool:
        return (
            self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4
            or self.header.block_type == wkw.Header.BLOCK_TYPE_LZ4HC
        )

    def _handle_compressed_write(
        self, absolute_offset: Tuple[int, int, int], data: np.ndarray
    ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        # calculate aligned bounding box
        file_bb = np.full(3, self.header.file_len * self.header.block_len)
        absolute_offset_np = np.array(absolute_offset)
        margin_to_top_left = absolute_offset_np % file_bb
        aligned_offset = absolute_offset_np - margin_to_top_left
        bottom_right = absolute_offset_np + np.array(data.shape[-3:])
        margin_to_bottom_right = (file_bb - (bottom_right % file_bb)) % file_bb
        aligned_bottom_right = bottom_right + margin_to_bottom_right
        aligned_shape = aligned_bottom_right - aligned_offset

        if (
            tuple(aligned_offset) != absolute_offset
            or tuple(aligned_shape) != data.shape[-3:]
        ):
            # the data is not aligned
            # read the aligned bounding box
            try:
                # We want to read the data at the absolute offset.
                # The absolute offset might be outside of the current view.
                # That is the case if the data is compressed but the view does not include the whole file on disk.
                # In this case we avoid checking the bounds because the aligned_offset and aligned_shape are calculated internally.
                aligned_data = self._read_without_checks(aligned_offset, aligned_shape)
            except AssertionError as e:
                raise AssertionError(
                    f"Writing compressed data failed. The compressed file is not fully inside the bounding box of the view (offset={self.global_offset}, size={self.size})."
                ) from e
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
            return cast(Tuple[int, int, int], tuple(aligned_offset)), aligned_data
        else:
            return absolute_offset, data

    def get_dtype(self) -> type:
        """
        Returns the dtype per channel of the data. For example `uint8`.
        """
        return self.header.voxel_type

    def __enter__(self) -> "View":
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self.close()

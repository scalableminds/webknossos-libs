import math
from concurrent.futures._base import Executor
from pathlib import Path
from types import TracebackType
from typing import Tuple, Optional, Type, Callable, Any, Union, List, cast

import cluster_tools
import numpy as np
from cluster_tools.schedulers.cluster_executor import ClusterExecutor
from wkw import Dataset, wkw

from wkcuber.api.TiffData.TiffMag import TiffMag, TiffMagHeader
from wkcuber.api.bounding_box import BoundingBox
from wkcuber.utils import wait_and_ensure_success


class View:
    def __init__(
        self,
        path_to_mag_dataset: str,
        header: Union[TiffMagHeader, wkw.Header],
        size: Tuple[int, int, int],
        global_offset: Tuple[int, int, int] = (0, 0, 0),
        is_bounded: bool = True,
    ):
        self.dataset: Optional[Dataset] = None
        self.path = path_to_mag_dataset
        self.header = header
        self.size = size
        self.global_offset = global_offset
        self.is_bounded = is_bounded
        self._is_opened = False

    def open(self) -> "View":
        pass

    def close(self) -> None:
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
        was_opened = self._is_opened
        # assert the size of the parameter data is not in conflict with the attribute self.size
        assert_non_negative_offset(relative_offset)
        self.assert_bounds(relative_offset, data.shape[-3:])

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
        was_opened = self._is_opened
        size = self.size if size is None else size

        # assert the parameter size is not in conflict with the attribute self.size
        self.assert_bounds(offset, size)

        # calculate the absolute offset
        absolute_offset = tuple(sum(x) for x in zip(self.global_offset, offset))

        if not was_opened:
            self.open()
        assert self.dataset is not None  # because the View was opened

        data = self.dataset.read(absolute_offset, size)

        if not was_opened:
            self.close()

        return data

    def get_view(
        self,
        size: Tuple[int, int, int],
        relative_offset: Tuple[int, int, int] = (0, 0, 0),
    ) -> "View":
        self.assert_bounds(relative_offset, size)
        view_offset = cast(
            Tuple[int, int, int], tuple(self.global_offset + np.array(relative_offset))
        )
        return type(self)(
            self.path,
            self.header,
            size=size,
            global_offset=view_offset,
            is_bounded=self.is_bounded,
        )

    def check_bounds(
        self, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> bool:
        for s1, s2, off in zip(self.size, size, offset):
            if s2 + off > s1 and self.is_bounded:
                return False
        return True

    def assert_bounds(
        self, offset: Tuple[int, int, int], size: Tuple[int, int, int]
    ) -> None:
        if not self.check_bounds(offset, size):
            raise AssertionError(
                f"Accessing data out of bounds: The passed parameter 'size' {size} exceeds the size of the current view ({self.size})"
            )

    def for_each_chunk(
        self,
        work_on_chunk: Callable[[Tuple["View", Tuple[Any, ...]]], None],
        job_args_per_chunk: Any,
        chunk_size: Tuple[int, int, int],
        executor: Union[ClusterExecutor, cluster_tools.WrappedProcessPoolExecutor],
    ) -> None:
        self._check_chunk_size(chunk_size)

        job_args = []

        for chunk in BoundingBox(self.global_offset, self.size).chunk(
            chunk_size, list(chunk_size)
        ):
            relative_offset = cast(
                Tuple[int, int, int],
                tuple(np.array(chunk.topleft) - np.array(self.global_offset)),
            )
            view = self.get_view(
                size=cast(Tuple[int, int, int], tuple(chunk.size)),
                relative_offset=relative_offset,
            )
            view.is_bounded = True
            job_args.append((view, job_args_per_chunk))

        # execute the work for each chunk
        wait_and_ensure_success(executor.map_to_futures(work_on_chunk, job_args))

    def _check_chunk_size(self, chunk_size: Tuple[int, int, int]) -> None:
        raise NotImplementedError

    def _is_compressed(self) -> bool:
        return False

    def _handle_compressed_write(
        self, absolute_offset: Tuple[int, int, int], data: np.ndarray
    ) -> Tuple[Tuple[int, int, int], np.ndarray]:
        return absolute_offset, data

    def __enter__(self) -> "View":
        return self

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self.close()


class WKView(View):
    header: wkw.Header

    def open(self) -> "WKView":
        if self._is_opened:
            raise Exception("Cannot open view: the view is already opened")
        else:
            self.dataset = Dataset.open(
                self.path
            )  # No need to pass the header to the wkw.Dataset
            self._is_opened = True
        return self

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
        margin_to_bottom_right = file_bb - (bottom_right % file_bb)
        aligned_bottom_right = bottom_right + margin_to_bottom_right
        aligned_shape = aligned_bottom_right - aligned_offset

        if (
            tuple(aligned_offset) != absolute_offset
            or tuple(aligned_shape) != data.shape[-3:]
        ):
            # the data is not aligned
            # read the aligned bounding box
            try:
                aligned_data = self.read(offset=aligned_offset, size=aligned_shape)
            except AssertionError as e:
                raise AssertionError(
                    f"Writing compressed data failed. The compressed file is not fully inside the bounding box of the view (offset={self.global_offset}, size={self.size}). "
                    + str(e)
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
            return cast(Tuple[int, int, int], tuple(aligned_offset)), aligned_data
        else:
            return absolute_offset, data


class TiffView(View):
    def open(self) -> "TiffView":
        if self._is_opened:
            raise Exception("Cannot open view: the view is already opened")
        else:
            self.dataset = TiffMag.open(self.path, self.header)
            self._is_opened = True
        return self

    def _check_chunk_size(self, chunk_size: Tuple[int, int, int]) -> None:
        assert chunk_size is not None

        if 0 in chunk_size:
            raise AssertionError(
                f"The passed parameter 'chunk_size' {chunk_size} contains at least one 0. This is not allowed."
            )

        if self.header.tile_size is None:
            # non tiled tiff dataset
            if self.size[0:2] != chunk_size[0:2]:
                raise AssertionError(
                    f"The x- and y-length of the passed parameter 'chunk_size' {chunk_size} do not match with the size of the view {self.size}."
                )
        else:
            # tiled tiff dataset
            file_dim = tuple(self.header.tile_size) + (
                1,
            )  # the z-dimension of an image is 1
            if (np.array(chunk_size) % file_dim).any():
                raise AssertionError(
                    f"The passed parameter 'chunk_size' {chunk_size} must be a multiple of the file size {file_dim}"
                )


def assert_non_negative_offset(offset: Tuple[int, int, int]) -> None:
    all_positive = all(i >= 0 for i in offset)
    if not all_positive:
        raise Exception(
            "All elements of the offset need to be positive: %s" % "("
            + ",".join(map(str, offset))
            + ")"
        )

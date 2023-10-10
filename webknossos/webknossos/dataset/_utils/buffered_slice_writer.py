import os
import traceback
import warnings
from logging import error, info
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, List, Optional, Type

import numpy as np
import psutil

from ...geometry import BoundingBox, Vec3Int, Vec3IntLike

if TYPE_CHECKING:
    from ..view import View


def log_memory_consumption(additional_output: str = "") -> None:
    pid = os.getpid()
    process = psutil.Process(pid)
    info(
        "Currently consuming {:.2f} GB of memory ({:.2f} GB still available) "
        "in process {}. {}".format(
            process.memory_info().rss / 1024**3,
            psutil.virtual_memory().available / 1024**3,
            pid,
            additional_output,
        )
    )


class BufferedSliceWriter:
    def __init__(
        self,
        view: "View",
        offset: Optional[Vec3IntLike] = None,
        # json_update_allowed enables the update of the bounding box and rewriting of the properties json.
        # It should be False when parallel access is intended.
        json_update_allowed: bool = True,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2,  # z
        *,
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        use_logging: bool = False,
    ) -> None:
        """see `View.get_buffered_slice_writer()`"""

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        self.use_logging = use_logging
        self.json_update_allowed = json_update_allowed
        if offset is None and relative_offset is None and absolute_offset is None:
            relative_offset = Vec3Int.zeros()
        if offset is not None:
            warnings.warn(
                "[DEPRECATION] Using offset for a buffered slice writer is deprecated. "
                + "Please use the parameter relative_offset or absolute_offset in Mag(1) instead.",
                DeprecationWarning,
            )
        self.offset = None if offset is None else Vec3Int(offset)
        self.relative_offset = (
            None if relative_offset is None else Vec3Int(relative_offset)
        )
        self.absolute_offset = (
            None if absolute_offset is None else Vec3Int(absolute_offset)
        )
        self.dimension = dimension

        effective_offset = Vec3Int.full(0)
        if self.relative_offset is not None:
            effective_offset = self.view.bounding_box.topleft + self.relative_offset

        if self.absolute_offset is not None:
            effective_offset = self.absolute_offset

        view_chunk_depth = self.view.info.chunk_shape[self.dimension]
        if (
            effective_offset is not None
            and effective_offset[self.dimension] % view_chunk_depth != 0
        ):
            warnings.warn(
                "[WARNING] Using an offset that doesn't align with the datataset's chunk size, "
                + "will slow down the buffered slice writer, because twice as many chunks will be written.",
            )
        if buffer_size >= view_chunk_depth and buffer_size % view_chunk_depth > 0:
            warnings.warn(
                "[WARNING] Using a buffer size that doesn't align with the datataset's chunk size, "
                + "will slow down the buffered slice writer.",
            )

        assert 0 <= dimension <= 2

        self.slices_to_write: List[np.ndarray] = []
        self.current_slice: Optional[int] = None
        self.buffer_start_slice: Optional[int] = None

    def _flush_buffer(self) -> None:
        if len(self.slices_to_write) == 0:
            return

        assert (
            len(self.slices_to_write) <= self.buffer_size
        ), "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."

        uniq_dtypes = set(map(lambda _slice: _slice.dtype, self.slices_to_write))
        assert (
            len(uniq_dtypes) == 1
        ), "The buffer of BufferedSliceWriter contains slices with differing dtype."
        assert uniq_dtypes.pop() == self.dtype, (
            "The buffer of BufferedSliceWriter contains slices with a dtype "
            "which differs from the dtype with which the BufferedSliceWriter was instantiated."
        )

        if self.use_logging:
            info(
                "({}) Writing {} slices at position {}.".format(
                    getpid(), len(self.slices_to_write), self.buffer_start_slice
                )
            )
            log_memory_consumption()

        try:
            assert (
                self.buffer_start_slice is not None
            ), "Failed to write buffer: The buffer_start_slice is not set."
            max_width = max(section.shape[-2] for section in self.slices_to_write)
            max_height = max(section.shape[-1] for section in self.slices_to_write)
            channel_count = self.slices_to_write[0].shape[0]

            buffer_bbox = BoundingBox(
                (0, 0, 0), (max_width, max_height, self.buffer_size)
            )

            shard_dimensions = self.view._get_file_dimensions().moveaxis(
                -1, self.dimension
            )
            chunk_size = Vec3Int(
                min(shard_dimensions[0], max_width),
                min(shard_dimensions[1], max_height),
                self.buffer_size,
            )
            for chunk_bbox in buffer_bbox.chunk(chunk_size):
                info(f"Writing chunk {chunk_bbox}")
                width, height, _ = chunk_bbox.size
                data = np.zeros(
                    (channel_count, width, height, self.buffer_size),
                    dtype=self.slices_to_write[0].dtype,
                )

                z = 0
                for section in self.slices_to_write:
                    section_chunk = section[
                        :,
                        chunk_bbox.topleft.x : chunk_bbox.bottomright.x,
                        chunk_bbox.topleft.y : chunk_bbox.bottomright.y,
                    ]
                    data[
                        :, 0 : section_chunk.shape[-2], 0 : section_chunk.shape[-1], z
                    ] = section_chunk

                    z += 1

                buffer_start = Vec3Int(
                    chunk_bbox.topleft.x, chunk_bbox.topleft.y, self.buffer_start_slice
                ).moveaxis(-1, self.dimension)
                buffer_start_mag1 = buffer_start * self.view.mag.to_vec3_int()

                data = np.moveaxis(data, -1, self.dimension + 1)

                self.view.write(
                    data,
                    offset=buffer_start.add_or_none(self.offset),
                    relative_offset=buffer_start_mag1.add_or_none(self.relative_offset),
                    absolute_offset=buffer_start_mag1.add_or_none(self.absolute_offset),
                    json_update_allowed=self.json_update_allowed,
                )
                del data

        except Exception as exc:
            error(
                "({}) An exception occurred in BufferedSliceWriter._flush_buffer with {} "
                "slices at position {}. Original error is:\n{}:{}\n\nTraceback:".format(
                    getpid(),
                    len(self.slices_to_write),
                    self.buffer_start_slice,
                    type(exc).__name__,
                    exc,
                )
            )
            traceback.print_tb(exc.__traceback__)
            error("\n")

            raise exc
        finally:
            self.slices_to_write = []

    def _get_slice_generator(self) -> Generator[None, np.ndarray, None]:
        current_slice = 0
        while True:
            data = yield  # Data gets send from the user
            if len(self.slices_to_write) == 0:
                self.buffer_start_slice = current_slice
            if len(data.shape) == 2:
                # The input data might contain channel data or not.
                # Bringing it into the same shape simplifies the code
                data = np.expand_dims(data, axis=0)
            self.slices_to_write.append(data)
            current_slice += 1

            if current_slice % self.buffer_size == 0:
                self._flush_buffer()

    def __enter__(self) -> Generator[None, np.ndarray, None]:
        gen = self._get_slice_generator()
        # It is necessary to start the generator by sending "None"
        gen.send(None)  # type: ignore
        return gen

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self._flush_buffer()

import logging
import os
import traceback
import warnings
from collections.abc import Generator
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
import psutil

from ...geometry import BoundingBox, NDBoundingBox, Vec3Int, Vec3IntLike

if TYPE_CHECKING:
    from ..view import View


logger = logging.getLogger(__name__)


def log_memory_consumption(additional_output: str = "") -> None:
    pid = os.getpid()
    process = psutil.Process(pid)
    logger.info(
        f"Currently consuming {process.memory_info().rss / 1024**3:.2f} GB of memory ({psutil.virtual_memory().available / 1024**3:.2f} GB still available) "
        f"in process {pid}. {additional_output}"
    )


class BufferedSliceWriter:
    def __init__(
        self,
        view: "View",
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2,  # z
        *,
        relative_offset: Vec3IntLike | None = None,  # in mag1
        absolute_offset: Vec3IntLike | None = None,  # in mag1
        relative_bounding_box: NDBoundingBox | None = None,  # in mag1
        absolute_bounding_box: NDBoundingBox | None = None,  # in mag1
        use_logging: bool = False,
        allow_unaligned: bool = False,
    ) -> None:
        """see `View.get_buffered_slice_writer()`"""

        self._view = view
        self._buffer_size = buffer_size
        self._dtype = self._view.get_dtype()
        self._use_logging = use_logging
        self._allow_unaligned = allow_unaligned
        self._bbox: NDBoundingBox
        self._slices_to_write: list[np.ndarray] = []
        self._current_slice: int | None = None
        self._buffer_start_slice: int | None = None

        self.reset_offset(
            relative_offset,
            absolute_offset,
            relative_bounding_box,
            absolute_bounding_box,
        )

        assert 0 <= dimension <= 2  # either x (0), y (1) or z (2)
        self.dimension = dimension

        view_shard_depth = self._view.info.chunk_shape[dimension]
        if (
            self._bbox is not None
            and self._bbox.topleft_xyz[self.dimension] % view_shard_depth != 0
        ):
            msg = (
                "Using an offset that doesn't align with the datataset's shard shape, "
                + "will slow down the buffered slice writer, because twice as many shards will be written. "
                + f"Got offset {self._bbox.topleft_xyz[self.dimension]} and shard depth {view_shard_depth}."
            )
            if allow_unaligned:
                warnings.warn("[WARNING] " + msg, category=UserWarning)
            else:
                raise ValueError(msg)
        if buffer_size >= view_shard_depth and buffer_size % view_shard_depth > 0:
            msg = (
                "Using a buffer size that doesn't align with the datataset's shard shape, "
                + "will slow down the buffered slice writer. "
                + f"Got buffer size {buffer_size} and shard depth {view_shard_depth}."
            )
            if allow_unaligned:
                warnings.warn("[WARNING] " + msg, category=UserWarning)
            else:
                raise ValueError(msg)

    def _flush_buffer(self) -> None:
        if len(self._slices_to_write) == 0:
            return

        assert len(self._slices_to_write) <= self._buffer_size, (
            "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."
        )

        uniq_dtypes = set(map(lambda _slice: _slice.dtype, self._slices_to_write))
        assert len(uniq_dtypes) == 1, (
            "The buffer of BufferedSliceWriter contains slices with differing dtype."
        )
        assert uniq_dtypes.pop() == self._dtype, (
            "The buffer of BufferedSliceWriter contains slices with a dtype "
            "which differs from the dtype with which the BufferedSliceWriter was instantiated."
        )

        if self._use_logging:
            logger.info(
                f"({getpid()}) Writing {len(self._slices_to_write)} slices at position {self._buffer_start_slice}."
            )
            log_memory_consumption()

        try:
            assert self._buffer_start_slice is not None, (
                "Failed to write buffer: The buffer_start_slice is not set."
            )
            max_width = max(section.shape[-2] for section in self._slices_to_write)
            max_height = max(section.shape[-1] for section in self._slices_to_write)
            channel_count = self._slices_to_write[0].shape[0]
            buffer_depth = min(self._buffer_size, len(self._slices_to_write))
            buffer_start = Vec3Int.zeros().with_replaced(
                self.dimension, self._buffer_start_slice
            )

            bbox = self._bbox.with_size_xyz(
                Vec3Int(max_width, max_height, buffer_depth).moveaxis(
                    -1, self.dimension
                )
            ).offset(buffer_start)

            shard_dimensions = self._view._get_file_dimensions()
            chunk_size = Vec3Int(
                min(shard_dimensions[0], max_width),
                min(shard_dimensions[1], max_height),
                buffer_depth,
            ).moveaxis(-1, self.dimension)
            for chunk_bbox in bbox.chunk(chunk_size):
                if self._use_logging:
                    logger.info(f"Writing chunk {chunk_bbox}.")

                data = np.zeros(
                    (channel_count, *chunk_bbox.size),
                    dtype=self._slices_to_write[0].dtype,
                )
                section_topleft = Vec3Int(
                    (chunk_bbox.topleft_xyz - bbox.topleft_xyz).moveaxis(
                        self.dimension, -1
                    )
                )
                section_bottomright = Vec3Int(
                    (chunk_bbox.bottomright_xyz - bbox.topleft_xyz).moveaxis(
                        self.dimension, -1
                    )
                )

                z_index = chunk_bbox.index_xyz[self.dimension]

                z = 0
                for section in self._slices_to_write:
                    section_chunk = section[
                        :,
                        section_topleft.x : section_bottomright.x,
                        section_topleft.y : section_bottomright.y,
                    ]
                    # Section chunk includes the axes c, x, y. The remaining axes are added by considering
                    # the length of the bbox. Since the bbox does not contain the channel, we subtract 2
                    # instead of 3.
                    section_chunk = section_chunk[
                        (slice(None), slice(None), slice(None))
                        + tuple(np.newaxis for _ in range(len(bbox) - 2))
                    ]
                    section_chunk = np.moveaxis(
                        section_chunk,
                        [1, 2],
                        bbox.index_xyz[: self.dimension]
                        + bbox.index_xyz[self.dimension + 1 :],
                    )

                    slice_tuple = (slice(None),) + tuple(
                        slice(0, min(size1, size2))
                        for size1, size2 in zip(
                            chunk_bbox.size, section_chunk.shape[1:]
                        )
                    )

                    data[
                        slice_tuple[:z_index]
                        + (slice(z, z + 1),)
                        + slice_tuple[z_index + 1 :]
                    ] = section_chunk

                    z += 1

                self._view.write(
                    data,
                    absolute_bounding_box=chunk_bbox.from_mag_to_mag1(self._view._mag),
                    allow_unaligned=self._allow_unaligned,
                )
                del data

        except Exception as exc:
            logger.error(
                f"({getpid()}) An exception occurred in BufferedSliceWriter._flush_buffer with {len(self._slices_to_write)} "
                f"slices at position {self._buffer_start_slice}. Original error is:\n{type(exc).__name__}:{exc}\n\nTraceback:"
            )
            traceback.print_tb(exc.__traceback__)
            logger.error("\n")

            raise exc
        finally:
            self._slices_to_write = []

    def _get_slice_generator(self) -> Generator[None, np.ndarray, None]:
        current_slice = 0
        while True:
            data = yield  # Data gets send from the user
            if len(self._slices_to_write) == 0:
                self._buffer_start_slice = current_slice
            if len(data.shape) == 2:
                # The input data might contain channel data or not.
                # Bringing it into the same shape simplifies the code
                data = np.expand_dims(data, axis=0)
            self._slices_to_write.append(data)
            current_slice += 1

            if current_slice % self._buffer_size == 0:
                self._flush_buffer()

    def send(self, value: np.ndarray) -> None:
        self._generator.send(value)

    def reset_offset(
        self,
        relative_offset: Vec3IntLike | None = None,  # in mag1
        absolute_offset: Vec3IntLike | None = None,  # in mag1
        relative_bounding_box: NDBoundingBox | None = None,  # in mag1
        absolute_bounding_box: NDBoundingBox | None = None,  # in mag1
    ) -> None:
        if self._slices_to_write:
            self._flush_buffer()

        # Reset the generator
        self._generator = self._get_slice_generator()
        next(self._generator)

        if (
            relative_offset is None
            and absolute_offset is None
            and relative_bounding_box is None
            and absolute_bounding_box is None
        ):
            relative_offset = Vec3Int.zeros()

        if relative_offset is not None:
            self._bbox = BoundingBox(
                self._view.bounding_box.topleft + relative_offset, Vec3Int.zeros()
            )

        if absolute_offset is not None:
            self._bbox = BoundingBox(absolute_offset, Vec3Int.zeros())

        if relative_bounding_box is not None:
            self._bbox = relative_bounding_box.offset(self._view.bounding_box.topleft)

        if absolute_bounding_box is not None:
            self._bbox = absolute_bounding_box

    def __enter__(self) -> "BufferedSliceWriter":
        self._generator = self._get_slice_generator()
        next(self._generator)
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self._flush_buffer()

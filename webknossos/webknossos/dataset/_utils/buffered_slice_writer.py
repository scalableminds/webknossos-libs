import os
import traceback
import warnings
from logging import error, info
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, List, Optional, Type

import numpy as np
import psutil

from webknossos.geometry.nd_bounding_box import NDBoundingBox

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
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        use_logging: bool = False,
    ) -> None:
        """see `View.get_buffered_slice_writer()`"""

        self._view = view
        self._buffer_size = buffer_size
        self._dtype = self._view.get_dtype()
        self._use_logging = use_logging
        self._json_update_allowed = json_update_allowed
        self._bbox: NDBoundingBox
        self._slices_to_write: List[np.ndarray] = []
        self._current_slice: Optional[int] = None
        self._buffer_start_slice: Optional[int] = None

        self.reset_offset(
            offset,
            relative_offset,
            absolute_offset,
            relative_bounding_box,
            absolute_bounding_box,
        )

        assert 0 <= dimension <= 2  # either x (0), y (1) or z (2)
        self.dimension = dimension

        view_chunk_depth = self._view.info.chunk_shape[dimension]
        if (
            self._bbox is not None
            and self._bbox.topleft_xyz[self.dimension] % view_chunk_depth != 0
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

    def _flush_buffer(self) -> None:
        if len(self._slices_to_write) == 0:
            return

        assert (
            len(self._slices_to_write) <= self._buffer_size
        ), "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."

        uniq_dtypes = set(map(lambda _slice: _slice.dtype, self._slices_to_write))
        assert (
            len(uniq_dtypes) == 1
        ), "The buffer of BufferedSliceWriter contains slices with differing dtype."
        assert uniq_dtypes.pop() == self._dtype, (
            "The buffer of BufferedSliceWriter contains slices with a dtype "
            "which differs from the dtype with which the BufferedSliceWriter was instantiated."
        )

        if self._use_logging:
            info(
                "({}) Writing {} slices at position {}.".format(
                    getpid(), len(self._slices_to_write), self._buffer_start_slice
                )
            )
            log_memory_consumption()

        try:
            assert (
                self._buffer_start_slice is not None
            ), "Failed to write buffer: The buffer_start_slice is not set."
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
                    info(f"Writing chunk {chunk_bbox}.")

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
                    json_update_allowed=self._json_update_allowed,
                    absolute_bounding_box=chunk_bbox.from_mag_to_mag1(self._view._mag),
                )
                del data

        except Exception as exc:
            error(
                "({}) An exception occurred in BufferedSliceWriter._flush_buffer with {} "
                "slices at position {}. Original error is:\n{}:{}\n\nTraceback:".format(
                    getpid(),
                    len(self._slices_to_write),
                    self._buffer_start_slice,
                    type(exc).__name__,
                    exc,
                )
            )
            traceback.print_tb(exc.__traceback__)
            error("\n")

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
        offset: Optional[Vec3IntLike] = None,  # deprecated, relative in current mag
        relative_offset: Optional[Vec3IntLike] = None,  # in mag1
        absolute_offset: Optional[Vec3IntLike] = None,  # in mag1
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
    ) -> None:
        if self._slices_to_write:
            self._flush_buffer()

        # Reset the generator
        self._generator = self._get_slice_generator()
        next(self._generator)

        if (
            offset is None
            and relative_offset is None
            and absolute_offset is None
            and relative_bounding_box is None
            and absolute_bounding_box is None
        ):
            relative_offset = Vec3Int.zeros()
        if offset is not None:
            warnings.warn(
                "[DEPRECATION] Using offset for a buffered slice writer is deprecated. "
                + "Please use the parameter relative_offset or absolute_offset in Mag(1) instead.",
                DeprecationWarning,
            )

        if offset is not None:
            self._bbox = BoundingBox(
                self._view.bounding_box.topleft_xyz + Vec3Int(offset) * self._view.mag,
                Vec3Int.zeros(),
            )

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
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        self._flush_buffer()

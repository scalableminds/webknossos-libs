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
    from ..layer.view import View


logger = logging.getLogger(__name__)


def log_memory_consumption(additional_output: str = "") -> None:
    pid = os.getpid()
    process = psutil.Process(pid)
    logger.info(
        f"Currently consuming {process.memory_info().rss / 1024**3:.2f} GB of memory ({psutil.virtual_memory().available / 1024**3:.2f} GB still available) "
        f"in process {pid}. {additional_output}"
    )


def _parse_dimension(dimension: str | int) -> str:
    if isinstance(dimension, int):
        # warnings.warn(
        #     f"[DEPRECATION] Setting `dimension` as an integer is deprecated, please use a named axis instead, e.g. `dimension='z'`. Got {dimension}.",
        #     DeprecationWarning,
        #     stacklevel=3,
        # )
        assert 0 <= dimension <= 2  # either x (0), y (1) or z (2)
        if dimension == 0:
            return "x"
        elif dimension == 1:
            return "y"
        else:
            return "z"
    else:
        assert dimension in ["x", "y", "z"]
        return dimension


class BufferedSliceWriter:
    def __init__(
        self,
        view: "View",
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: str | int = "z",  # z
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

        self.dimension = _parse_dimension(dimension)

        self.reset_offset(
            relative_offset,
            absolute_offset,
            relative_bounding_box,
            absolute_bounding_box,
        )

        view_shard_depth = self._view.info.chunk_shape.get(self.dimension)
        if (
            self._bbox is not None
            and self._bbox.topleft.get(self.dimension) % view_shard_depth != 0
        ):
            msg = (
                "Using an offset that doesn't align with the datataset's shard shape, "
                + "will slow down the buffered slice writer, because twice as many shards will be written. "
                + f"Got offset {self._bbox.topleft.get(self.dimension)} and shard depth {view_shard_depth}."
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

    def _get_other_axes(self) -> tuple[str, str]:
        if self.dimension == "x":
            width_axis = "y"
            height_axis = "z"
        elif self.dimension == "y":
            width_axis = "x"
            height_axis = "z"
        else:
            width_axis = "x"
            height_axis = "y"
        return width_axis, height_axis

    def _flush_buffer(self) -> None:
        width_axis, height_axis = self._get_other_axes()

        if len(self._slices_to_write) == 0:
            return

        assert len(self._slices_to_write) <= self._buffer_size, (
            "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."
        )
        assert all(len(s.shape) == len(self._bbox) for s in self._slices_to_write), (
            "The slices in the buffer are not 3D."
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
            max_width = max(
                section.shape[self._bbox.axes.index(width_axis)]
                for section in self._slices_to_write
            )
            max_height = max(
                section.shape[self._bbox.axes.index(height_axis)]
                for section in self._slices_to_write
            )
            buffer_depth = min(self._buffer_size, len(self._slices_to_write))

            bbox = (
                self._bbox.with_bounds(
                    self.dimension,
                    new_topleft=self._bbox.topleft.get(self.dimension)
                    + self._buffer_start_slice,
                    new_size=buffer_depth,
                )
                .with_bounds(width_axis, new_size=max_width)
                .with_bounds(height_axis, new_size=max_height)
            )

            shard_shape = self._view.info.shard_shape
            chunk_shape = (
                shard_shape.with_replaced(self.dimension, buffer_depth)
                .with_replaced(width_axis, min(shard_shape.get(width_axis), max_width))
                .with_replaced(
                    height_axis, min(shard_shape.get(height_axis), max_height)
                )
            )

            for chunk_bbox in bbox.chunk(chunk_shape):
                if self._use_logging:
                    logger.info(f"Writing chunk {chunk_bbox}.")

                data = np.zeros(
                    chunk_bbox.size.to_tuple(),
                    dtype=self._slices_to_write[0].dtype,
                )

                section_selector_list: list[slice] = []
                for axis in bbox.axes:
                    if axis == width_axis:
                        section_selector_list.append(
                            slice(
                                chunk_bbox.topleft.get(width_axis)
                                - bbox.topleft.get(width_axis),
                                chunk_bbox.bottomright.get(width_axis)
                                - bbox.topleft.get(width_axis),
                            )
                        )
                    elif axis == height_axis:
                        section_selector_list.append(
                            slice(
                                chunk_bbox.topleft.get(height_axis)
                                - bbox.topleft.get(height_axis),
                                chunk_bbox.bottomright.get(height_axis)
                                - bbox.topleft.get(height_axis),
                            )
                        )
                    else:
                        section_selector_list.append(slice(None))
                section_selector = tuple(section_selector_list)

                z_index = chunk_bbox.axes.index(self.dimension)

                z = 0
                for section in self._slices_to_write:
                    section_chunk = section[section_selector]

                    data_selector = tuple(
                        slice(0, min(size1, size2))
                        for size1, size2 in zip(chunk_bbox.size, section_chunk.shape)
                    )
                    data_selector = (
                        data_selector[:z_index]
                        + (slice(z, z + 1),)
                        + data_selector[z_index + 1 :]
                    )
                    data[data_selector] = section_chunk
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
        width_axis, height_axis = self._get_other_axes()
        while True:
            data = yield  # Data gets send from the user
            if len(self._slices_to_write) == 0:
                self._buffer_start_slice = current_slice
            if len(data.shape) == 2:
                axis_map = {width_axis: 0, height_axis: 1}
            elif len(data.shape) == 3:
                assert "c" in self._bbox.axes
                axis_map = {"c": 0, width_axis: 1, height_axis: 2}
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")

            # The order of the axes in the input data might not match the order of the axes in the bbox
            data = np.transpose(
                data, [axis_map[a] for a in self._bbox.axes if a in axis_map]
            )
            # Add missing axes to the data
            for i, a in enumerate(self._bbox.axes):
                if a not in axis_map:
                    data = np.expand_dims(data, axis=i)

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

        bbox: NDBoundingBox
        if relative_offset is not None:
            bbox = BoundingBox(
                self._view.bounding_box.topleft + relative_offset, Vec3Int.zeros()
            )

        elif absolute_offset is not None:
            bbox = BoundingBox(absolute_offset, Vec3Int.zeros()).normalize_axes(
                self._view.info.num_channels
            )

        elif relative_bounding_box is not None:
            bbox = relative_bounding_box.offset(self._view.bounding_box.topleft)

        elif absolute_bounding_box is not None:
            bbox = absolute_bounding_box

        else:
            raise ValueError("No bounding box specified.")

        self._bbox = bbox.normalize_axes(self._view.info.num_channels)

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

import logging
from collections.abc import Generator
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from ...geometry import NDBoundingBox
from .buffered_slice_writer import _parse_dimension

if TYPE_CHECKING:
    from ..layer.view import View

logger = logging.getLogger(__name__)


class BufferedSliceReader:
    def __init__(
        self,
        view: "View",
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: str | int = "z",
        *,
        relative_bounding_box: NDBoundingBox | None = None,  # in mag1
        absolute_bounding_box: NDBoundingBox | None = None,  # in mag1
        use_logging: bool = False,
    ) -> None:
        """see `View.get_buffered_slice_reader()`"""

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        self.dimension = _parse_dimension(dimension)
        self.use_logging = use_logging

        if relative_bounding_box is None and absolute_bounding_box is None:
            absolute_bounding_box = view.bounding_box
        if relative_bounding_box is not None:
            assert absolute_bounding_box is None
            absolute_bounding_box = relative_bounding_box.offset(
                view.bounding_box.topleft
            )

        assert absolute_bounding_box is not None
        self.bbox_current_mag = absolute_bounding_box.in_mag(view.mag).normalize_axes(
            view.info.num_channels
        )

    def _get_slice_generator(self) -> Generator[np.ndarray, None, None]:
        chunk_shape = self.bbox_current_mag.size_xyz.with_replaced(
            self.dimension, self.buffer_size
        )

        for chunk_bbox in self.bbox_current_mag.chunk(chunk_shape):
            if self.use_logging:
                logger.info(f"({getpid()}) Reading data from bbox {chunk_bbox}.")
            data = self.view.read(
                absolute_bounding_box=chunk_bbox.from_mag_to_mag1(self.view.mag)
            )

            yield from np.rollaxis(data, chunk_bbox.axes.index(self.dimension))

    def __enter__(self) -> Generator[np.ndarray, None, None]:
        return self._get_slice_generator()

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _tb: TracebackType | None,
    ) -> None: ...

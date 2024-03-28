import warnings
from logging import info
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, Optional, Type

import numpy as np

if TYPE_CHECKING:
    from ..view import View

from ...geometry import BoundingBox, NDBoundingBox, Vec3IntLike


class BufferedSliceReader:
    def __init__(
        self,
        view: "View",
        offset: Optional[Vec3IntLike] = None,
        size: Optional[Vec3IntLike] = None,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2,  # z
        *,
        relative_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[NDBoundingBox] = None,  # in mag1
        use_logging: bool = False,
    ) -> None:
        """see `View.get_buffered_slice_reader()`"""

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        assert 0 <= dimension <= 2
        self.dimension = dimension
        self.use_logging = use_logging
        if offset is not None and size is not None:
            warnings.warn(
                "[DEPRECATION] Using offset and size for a buffered slice reader is deprecated. "
                + "Please use the parameter relative_bounding_box or absolute_bounding_box in Mag(1) instead.",
                DeprecationWarning,
            )
            assert relative_bounding_box is None and absolute_bounding_box is None
            absolute_bounding_box = BoundingBox(offset, size).from_mag_to_mag1(view.mag)
            offset = None
            size = None

        assert (
            offset is None and size is None
        ), "You have to set both offset and size or none of both."
        if relative_bounding_box is None and absolute_bounding_box is None:
            absolute_bounding_box = view.bounding_box
        if relative_bounding_box is not None:
            assert absolute_bounding_box is None
            absolute_bounding_box = relative_bounding_box.offset(
                view.bounding_box.topleft
            )

        assert absolute_bounding_box is not None
        self.bbox_current_mag = absolute_bounding_box.in_mag(view.mag)

    def _get_slice_generator(self) -> Generator[np.ndarray, None, None]:
        chunk_size = self.bbox_current_mag.size_xyz.to_list()
        chunk_size[self.dimension] = self.buffer_size

        for chunk in self.bbox_current_mag.chunk(chunk_size):
            if self.use_logging:
                info(f"({getpid()}) Reading data from bbox {chunk}.")
            data = self.view.read(
                absolute_bounding_box=chunk.from_mag_to_mag1(self.view.mag)
            )

            for current_slice in np.rollaxis(data, chunk.index_xyz[self.dimension]):
                yield current_slice

    def __enter__(self) -> Generator[np.ndarray, None, None]:
        return self._get_slice_generator()

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None: ...

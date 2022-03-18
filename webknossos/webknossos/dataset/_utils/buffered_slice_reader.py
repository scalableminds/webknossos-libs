import logging
import warnings
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, Optional, Type

import numpy as np

if TYPE_CHECKING:
    from webknossos.dataset import View

from webknossos.geometry import BoundingBox, Vec3IntLike
from webknossos.utils import get_chunks


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
        relative_bounding_box: Optional[BoundingBox] = None,  # in mag1
        absolute_bounding_box: Optional[BoundingBox] = None,  # in mag1
    ) -> None:
        """see `View.get_buffered_slice_reader()`"""

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        assert 0 <= dimension <= 2
        self.dimension = dimension
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
        for batch in get_chunks(
            list(
                range(
                    self.bbox_current_mag.topleft[self.dimension],
                    self.bbox_current_mag.bottomright[self.dimension],
                )
            ),
            self.buffer_size,
        ):
            n_slices = len(batch)
            batch_start_idx = batch[0]

            assert (
                n_slices <= self.buffer_size
            ), f"n_slices should at most be batch_size, but {n_slices} > {self.buffer_size}"

            bbox_offset = self.bbox_current_mag.topleft
            bbox_size = self.bbox_current_mag.size

            buffer_bounding_box = BoundingBox.from_tuple2(
                (
                    bbox_offset[: self.dimension]
                    + (batch_start_idx,)
                    + bbox_offset[self.dimension + 1 :],
                    bbox_size[: self.dimension]
                    + (n_slices,)
                    + bbox_size[self.dimension + 1 :],
                )
            )

            logging.debug(
                f"({getpid()}) Reading {n_slices} slices at position {batch_start_idx}."
            )
            data = self.view.read(
                absolute_bounding_box=buffer_bounding_box.from_mag_to_mag1(
                    self.view.mag
                )
            )

            for current_slice in np.rollaxis(
                data, self.dimension + 1
            ):  # The '+1' is important because the first dimension is the channel
                yield current_slice

    def __enter__(self) -> Generator[np.ndarray, None, None]:
        return self._get_slice_generator()

    def __exit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        ...

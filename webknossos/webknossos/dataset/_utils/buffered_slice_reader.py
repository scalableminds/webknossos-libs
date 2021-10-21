import logging
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, Optional, Tuple, Type, cast

import numpy as np

if TYPE_CHECKING:
    from webknossos.dataset import View

from webknossos.geometry import BoundingBox, Vec3Int
from webknossos.utils import get_chunks


class BufferedSliceReader(object):
    def __init__(
        self,
        view: "View",
        offset: Vec3Int,
        size: Vec3Int,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2,  # z
    ) -> None:
        """
        view : datasource
        offset : specifies the offset of the data to read (relative to the `view`)
        size : specifies the size of the data to read
        buffer_size : the number of slices that are read at once
        dimension : specifies along which axis the data is sliced (0=x; 1=y; 2=z)

        The size and offset are in the magnification of the `view`.
        """

        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        assert 0 <= dimension <= 2
        self.dimension = dimension
        bounding_box = BoundingBox(view.global_offset, view.size)
        self.target_bbox = bounding_box.intersected_with(
            BoundingBox(view.global_offset, size).offset(
                cast(Tuple[int, int, int], tuple(offset))
            )
        )

    def _get_slice_generator(self) -> Generator[np.ndarray, None, None]:
        for batch in get_chunks(
            list(
                range(
                    self.target_bbox.topleft[self.dimension],
                    self.target_bbox.bottomright[self.dimension],
                )
            ),
            self.buffer_size,
        ):
            n_slices = len(batch)
            batch_start_idx = batch[0]

            assert (
                n_slices <= self.buffer_size
            ), f"n_slices should at most be batch_size, but {n_slices} > {self.buffer_size}"

            bbox_offset = self.target_bbox.topleft
            bbox_size = self.target_bbox.size

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
            negative_view_offset = cast(
                Tuple[int, int, int], tuple([-o for o in self.view.global_offset])
            )  # this needs to be subtracted from the buffer_bounding_box because the view expects a relative offset
            data = self.view.read_bbox(buffer_bounding_box.offset(negative_view_offset))

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

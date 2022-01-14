import logging
import os
import traceback
from os import getpid
from types import TracebackType
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple, Type, cast

import numpy as np
import psutil

from webknossos.geometry import Vec3Int, Vec3IntLike

if TYPE_CHECKING:
    from webknossos.dataset import View


def log_memory_consumption(additional_output: str = "") -> None:
    pid = os.getpid()
    process = psutil.Process(pid)
    logging.info(
        "Currently consuming {:.2f} GB of memory ({:.2f} GB still available) "
        "in process {}. {}".format(
            process.memory_info().rss / 1024 ** 3,
            psutil.virtual_memory().available / 1024 ** 3,
            pid,
            additional_output,
        )
    )


class BufferedSliceWriter:
    def __init__(
        self,
        view: "View",
        offset: Vec3IntLike,
        # buffer_size specifies, how many slices should be aggregated until they are flushed.
        buffer_size: int = 32,
        dimension: int = 2,  # z
    ) -> None:
        """
        view : datasource
        offset : specifies the offset of the data to write (relative to the `view`)
        buffer_size : the number of slices that are read at once
        dimension : specifies along which axis the data is sliced (0=x; 1=y; 2=z)

        The size is in the magnification of the `view`.
        """
        self.view = view
        self.buffer_size = buffer_size
        self.dtype = self.view.get_dtype()
        self.offset = Vec3Int(offset)
        self.dimension = dimension

        assert 0 <= dimension <= 2

        self.buffer: List[np.ndarray] = []
        self.current_slice: Optional[int] = None
        self.buffer_start_slice: Optional[int] = None

    def _write_buffer(self) -> None:
        if len(self.buffer) == 0:
            return

        assert (
            len(self.buffer) <= self.buffer_size
        ), "The WKW buffer is larger than the defined batch_size. The buffer should have been flushed earlier. This is probably a bug in the BufferedSliceWriter."

        uniq_dtypes = set(map(lambda _slice: _slice.dtype, self.buffer))
        assert (
            len(uniq_dtypes) == 1
        ), "The buffer of BufferedSliceWriter contains slices with differing dtype."
        assert uniq_dtypes.pop() == self.dtype, (
            "The buffer of BufferedSliceWriter contains slices with a dtype "
            "which differs from the dtype with which the BufferedSliceWriter was instantiated."
        )

        logging.debug(
            "({}) Writing {} slices at position {}.".format(
                getpid(), len(self.buffer), self.buffer_start_slice
            )
        )
        log_memory_consumption()

        try:
            assert (
                self.buffer_start_slice is not None
            ), "Failed to write buffer: The buffer_start_slice is not set."
            buffer_start = [0, 0, 0]
            buffer_start[self.dimension] = self.buffer_start_slice
            offset = cast(
                Tuple[int, int, int],
                tuple(
                    [off + buff_off for off, buff_off in zip(self.offset, buffer_start)]
                ),
            )
            max_width = max(slice.shape[-2] for slice in self.buffer)
            max_height = max(slice.shape[-1] for slice in self.buffer)

            self.buffer = [
                np.pad(
                    slice,
                    mode="constant",
                    pad_width=[
                        (0, 0),
                        (0, max_width - slice.shape[-2]),
                        (0, max_height - slice.shape[-1]),
                    ],
                )
                for slice in self.buffer
            ]

            data = np.concatenate(
                [np.expand_dims(slice, self.dimension + 1) for slice in self.buffer],
                axis=self.dimension + 1,
            )
            self.view.write(data, offset)

        except Exception as exc:
            logging.error(
                "({}) An exception occurred in BufferedSliceWriter._write_buffer with {} "
                "slices at position {}. Original error is:\n{}:{}\n\nTraceback:".format(
                    getpid(),
                    len(self.buffer),
                    self.buffer_start_slice,
                    type(exc).__name__,
                    exc,
                )
            )
            traceback.print_tb(exc.__traceback__)
            logging.error("\n")

            raise exc
        finally:
            self.buffer = []

    def _get_slice_generator(self) -> Generator[None, np.ndarray, None]:
        current_slice = 0
        while True:
            data = yield  # Data gets send from the user
            if len(self.buffer) == 0:
                self.buffer_start_slice = current_slice
            if len(data.shape) == 2:
                # The input data might contain channel data or not.
                # Bringing it into the same shape simplifies the code
                data = np.expand_dims(data, axis=0)
            self.buffer.append(data)
            current_slice += 1

            if current_slice % self.buffer_size == 0:
                self._write_buffer()

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
        self._write_buffer()

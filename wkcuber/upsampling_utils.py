import logging
import math
from itertools import product
from typing import Tuple, List, cast

import numpy as np

from wkcuber.api.View import View
from wkcuber.utils import time_start, time_stop


def upsample_cube(cube_buffer: np.ndarray, factors: List[int]) -> np.ndarray:
    ds = cube_buffer.shape
    out_buf = np.zeros(tuple(s * f for s, f in zip(ds, factors)), cube_buffer.dtype)
    for dx in (0, factors[0] - 1):
        for dy in (0, factors[1] - 1):
            for dz in (0, factors[2] - 1):
                out_buf[
                    dx : out_buf.shape[0] : factors[0],
                    dy : out_buf.shape[1] : factors[1],
                    dz : out_buf.shape[2] : factors[2],
                ] = cube_buffer
    return out_buf


def upsample_cube_job(
    args: Tuple[View, View, int],
    mag_factors: List[float],
    buffer_edge_len: int,
    compress: bool,
    job_count_per_log: int,
) -> None:
    (source_view, target_view, i) = args
    use_logging = i % job_count_per_log == 0

    assert all(
        1 >= f for f in mag_factors
    ), f"mag_factors ({mag_factors}) for upsampling must be smaller than 1"

    if use_logging:
        logging.info(f"Upsampling of {target_view.global_offset}")

    try:
        if use_logging:
            time_start(f"Upsampling of {target_view.global_offset}")

        num_channels = target_view.header.num_channels
        shape = (num_channels,) + tuple(target_view.size)
        file_buffer = np.zeros(shape, target_view.get_dtype())

        tiles = product(
            *list(
                [
                    list(range(0, math.ceil(len)))
                    for len in np.array(target_view.size) / buffer_edge_len
                ]
            )
        )

        source_view.open()

        for tile in tiles:
            target_offset = np.array(tile) * buffer_edge_len
            source_offset = (mag_factors * target_offset).astype(int)
            source_size = cast(
                Tuple[int, int, int],
                tuple(
                    [
                        int(min(a, b))
                        for a, b in zip(
                            np.array(mag_factors) * buffer_edge_len,
                            source_view.size - source_offset,
                        )
                    ]
                ),
            )

            cube_buffer_channels = source_view.read(source_offset, source_size)

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    # Upsample the buffer
                    inverse_factors = [int(1 / f) for f in mag_factors]
                    data_cube = upsample_cube(cube_buffer, inverse_factors)

                    buffer_offset = target_offset
                    buffer_end = buffer_offset + data_cube.shape

                    file_buffer[
                        channel_index,
                        buffer_offset[0] : buffer_end[0],
                        buffer_offset[1] : buffer_end[1],
                        buffer_offset[2] : buffer_end[2],
                    ] = data_cube

        source_view.close()
        # Write the upsampled buffer to target
        if source_view.header.num_channels == 1:
            file_buffer = file_buffer[0]  # remove channel dimension
        target_view.write(file_buffer, allow_compressed_write=compress)
        if use_logging:
            time_stop(f"Upsampling of {target_view.global_offset}")

    except Exception as exc:
        logging.error(f"Upsampling of {target_view.global_offset} failed with {exc}")
        raise exc

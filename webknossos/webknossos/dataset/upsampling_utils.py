import logging
import math
from itertools import product
from typing import List, Tuple

import numpy as np

from webknossos.geometry import BoundingBox, Vec3Int
from webknossos.utils import time_start, time_stop

from .view import View


def _vec3int_mulf(vec: Vec3Int, factors: List[float]) -> Vec3Int:
    return Vec3Int(
        int(vec.x * factors[0]), int(vec.y * factors[1]), int(vec.z * factors[2])
    )


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
    buffer_shape: Vec3Int,
) -> None:
    (source_view, target_view, _i) = args

    assert all(
        1 >= f for f in mag_factors
    ), f"mag_factors ({mag_factors}) for upsampling must be smaller than 1"

    try:
        time_start(f"Upsampling of {target_view.bounding_box}")
        num_channels = target_view.info.num_channels
        target_size = target_view.bounding_box.in_mag(target_view.mag).size
        shape = (num_channels,) + target_size.to_tuple()
        file_buffer = np.zeros(shape, target_view.get_dtype())

        tiles = product(
            *list(
                [
                    list(range(0, math.ceil(len)))
                    for len in target_size.to_np() / buffer_shape.to_np()
                ]
            )
        )

        for tile in tiles:
            target_offset = Vec3Int(tile) * buffer_shape
            source_offset = _vec3int_mulf(target_offset, mag_factors)
            source_size = source_view.bounding_box.in_mag(source_view.mag).size
            source_size = _vec3int_mulf(buffer_shape, mag_factors).pairmin(
                source_size - source_offset
            )

            bbox = BoundingBox(source_offset, source_size)
            cube_buffer_channels = source_view.read(
                relative_bounding_box=bbox.from_mag_to_mag1(source_view.mag),
            )

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

        # Write the upsampled buffer to target
        if source_view.info.num_channels == 1:
            file_buffer = file_buffer[0]  # remove channel dimension
        target_view.write(file_buffer)
        time_stop(f"Upsampling of {target_view.bounding_box}")

    except Exception as exc:
        logging.error(
            f"Upsampling of target {target_view.bounding_box} failed with {exc}"
        )
        raise exc

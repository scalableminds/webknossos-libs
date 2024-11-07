import logging
import math
from itertools import product
from typing import List, Tuple

import numpy as np

from ..geometry import Vec3Int
from .view import View


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
        num_channels = target_view.info.num_channels
        target_bbox_in_mag = target_view.bounding_box.in_mag(target_view.mag)
        shape = (num_channels,) + target_bbox_in_mag.size.to_tuple()
        shape_xyz = target_bbox_in_mag.size_xyz
        file_buffer = np.zeros(shape, target_view.get_dtype())

        tiles = product(
            *list(
                [
                    list(range(0, math.ceil(length)))
                    for length in shape_xyz.to_np() / buffer_shape.to_np()
                ]
            )
        )

        for tile in tiles:
            target_offset = Vec3Int(tile) * buffer_shape
            source_offset = target_offset * source_view.mag
            source_size = source_view.bounding_box.size_xyz
            source_size = (buffer_shape * source_view.mag).pairmin(
                source_size - source_offset
            )

            bbox = source_view.bounding_box.offset(source_offset).with_size_xyz(
                source_size
            )
            cube_buffer_channels = source_view.read_xyz(
                absolute_bounding_box=bbox,
            )

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    # Upsample the buffer
                    inverse_factors = [int(1 / f) for f in mag_factors]
                    data_cube = upsample_cube(cube_buffer, inverse_factors)

                    buffer_bbox = target_view.bounding_box.with_topleft_xyz(
                        target_offset * inverse_factors
                    ).with_size_xyz(data_cube.shape)
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)
                    file_buffer[(channel_index,) + buffer_bbox.to_slices_xyz()] = (
                        data_cube
                    )

        target_view.write(file_buffer, absolute_bounding_box=target_view.bounding_box)

    except Exception as exc:
        logging.error(
            f"Upsampling of target {target_view.bounding_box} failed with {exc}"
        )
        raise exc

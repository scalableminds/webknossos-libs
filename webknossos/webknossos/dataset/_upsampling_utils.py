import logging
import math
from itertools import product

import numpy as np

from ..geometry import Vec3Int
from .view import View


def upsample_cube(cube_buffer: np.ndarray, factors: list[int]) -> np.ndarray:
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
    args: tuple[View, View, int],
    mag_factors: list[float],
    buffer_shape: Vec3Int,
) -> None:
    (source_view, target_view, _i) = args

    assert all(1 >= f for f in mag_factors), (
        f"mag_factors ({mag_factors}) for upsampling must be smaller than 1"
    )
    inverse_factors = [int(1 / f) for f in mag_factors]

    try:
        num_channels = target_view.info.num_channels
        target_bbox = target_view.bounding_box.align_with_mag(
            target_view.info.shard_shape, True
        )
        shape_xyz = target_bbox.size // target_view.mag

        tiles = product(
            *list(
                [
                    list(range(0, math.ceil(length)))
                    for length in shape_xyz.to_np() / buffer_shape.to_np()
                ]
            )
        )

        for tile in tiles:
            tile_bbox = target_view.bounding_box.offset(
                Vec3Int(tile) * buffer_shape * target_view.mag
            ).with_size_xyz(buffer_shape * target_view.mag)

            bbox = target_bbox.intersected_with(tile_bbox)
            shape = (num_channels,) + (
                bbox.in_mag(target_view.mag)
            ).size.to_tuple()  # shape of file buffer in target mag
            file_buffer = np.zeros(
                shape, dtype=target_view.get_dtype()
            )  # zero filled file buffer
            cube_buffer_channels = source_view.read_xyz(
                absolute_bounding_box=bbox,
            )  # read from source view to receive data in shape of file buffer * mag_factors

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    # Upsample the buffer
                    data_cube = upsample_cube(cube_buffer, inverse_factors)

                    buffer_bbox = target_view.bounding_box.with_topleft_xyz(
                        source_offset * inverse_factors
                    ).with_size_xyz(data_cube.shape)
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)
                    file_buffer[(channel_index,) + buffer_bbox.to_slices_xyz()] = (
                        data_cube
                    )

            target_view.write(file_buffer, absolute_bounding_box=bbox)

    except Exception as exc:
        logging.error(
            f"Upsampling of target {target_view.bounding_box} failed with {exc}"
        )
        raise exc

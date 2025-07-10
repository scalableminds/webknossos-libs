import logging

import numpy as np

from ..geometry import Vec3Int
from .data_format import DataFormat
from .view import View

logger = logging.getLogger(__name__)


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
    if (
        target_view._data_format != DataFormat.WKW
        and target_view._is_compressed() == False
    ):
        assert buffer_shape % target_view.info.shard_shape == Vec3Int.zeros(), (
            f"buffer_shape ({buffer_shape}) must be divisible by shard_shape ({target_view.info.shard_shape})"
        )
    inverse_factors = [int(1 / f) for f in mag_factors]

    try:
        num_channels = target_view.info.num_channels

        for chunk in target_view.bounding_box.chunk(
            buffer_shape * target_view.mag, buffer_shape * target_view.mag
        ):
            shape = (num_channels,) + (chunk.in_mag(target_view.mag)).size.to_tuple()
            file_buffer = np.zeros(shape, dtype=target_view.get_dtype())
            cube_buffer_channels = source_view.read_xyz(
                absolute_bounding_box=chunk,
            )

            for channel_index in range(num_channels):
                cube_buffer = cube_buffer_channels[channel_index]

                if not np.all(cube_buffer == 0):
                    data_cube = upsample_cube(cube_buffer, inverse_factors)

                    buffer_bbox = chunk.with_topleft_xyz(Vec3Int.zeros()).with_size_xyz(
                        data_cube.shape
                    )
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)
                    file_buffer[(channel_index,) + buffer_bbox.to_slices_xyz()] = (
                        data_cube
                    )

            target_view.write(file_buffer, absolute_bounding_box=chunk)

    except Exception as exc:
        logger.error(
            f"Upsampling of target {target_view.bounding_box} failed with {exc}"
        )
        raise exc

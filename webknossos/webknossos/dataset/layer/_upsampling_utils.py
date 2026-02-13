import logging

import numpy as np

from webknossos.dataset.layer.view import View
from webknossos.dataset_properties import DataFormat
from webknossos.geometry import Vec3Int

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

    target_bbox = target_view.bounding_box.normalize_axes(target_view.info.num_channels)

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

        for chunk in target_bbox.chunk(
            buffer_shape * target_view.mag, buffer_shape * target_view.mag
        ):
            shape = chunk.in_mag(target_view.mag).size.to_tuple()
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
                    if "c" in buffer_bbox.axes:
                        buffer_bbox = buffer_bbox.with_bounds("c", new_size=1)
                    data_cube = buffer_bbox.xyz_array_to_bbox_shape(data_cube)
                    slices: list[int | slice] = list(buffer_bbox.to_slices_xyz())
                    if "c" in buffer_bbox.axes:
                        slices[buffer_bbox.index.c] = channel_index
                    file_buffer[tuple(slices)] = data_cube

            target_view.write(file_buffer, absolute_bounding_box=chunk)

    except Exception as exc:
        logger.error(f"Upsampling of target {target_bbox} failed with {exc}")
        raise exc

import numpy as np
from wkcuber.downsampling import (
    InterpolationModes,
    downsample_cube,
    downsample_cube_job,
    cube_addresses,
)
from wkcuber.utils import WkwDatasetInfo, open_wkw

WKW_CUBE_SIZE = 1024
CUBE_EDGE_LEN = 256

source_info = WkwDatasetInfo("testdata/WT1_wkw", "color", "uint8", 1)
target_info = WkwDatasetInfo("testoutput/WT1_wkw", "color", "uint8", 2)


def read_wkw(wkw_info, offset, size):
    with open_wkw(wkw_info) as wkw_dataset:
        return wkw_dataset.read(offset, size)[0]


def test_downsample_cube():
    buffer = np.zeros((CUBE_EDGE_LEN,) * 3, dtype=np.uint8)
    buffer[:, :, :] = np.arange(0, CUBE_EDGE_LEN)

    output = downsample_cube(buffer, 2, InterpolationModes.MEDIAN)

    assert output.shape == (CUBE_EDGE_LEN // 2,) * 3
    assert buffer[0, 0, 0] == 0
    assert buffer[0, 0, 1] == 1
    assert np.all(output[:, :, :] == np.arange(0, CUBE_EDGE_LEN, 2))


def test_cube_addresses():
    cubes_per_file = WKW_CUBE_SIZE // CUBE_EDGE_LEN

    addresses = cube_addresses(source_info, CUBE_EDGE_LEN)
    assert len(addresses) == (5 * cubes_per_file) * (5 * cubes_per_file) * (
        1 * cubes_per_file
    )
    assert min(addresses) == (0, 0, 0)
    assert max(addresses) == (
        5 * cubes_per_file - 1,
        5 * cubes_per_file - 1,
        1 * cubes_per_file - 1,
    )


def test_downsample_cube_job():
    offset = (3, 3, 0)
    downsample_cube_job(
        source_info, target_info, 2, InterpolationModes.MAX, CUBE_EDGE_LEN, offset
    )

    source_buffer = read_wkw(
        source_info,
        tuple(a * CUBE_EDGE_LEN * 2 for a in offset),
        (CUBE_EDGE_LEN * 2,) * 3,
    )
    assert np.any(source_buffer != 0)

    target_buffer = read_wkw(
        target_info, tuple(a * CUBE_EDGE_LEN for a in offset), (CUBE_EDGE_LEN,) * 3
    )
    assert np.any(target_buffer != 0)

    assert np.all(
        target_buffer == downsample_cube(source_buffer, 2, InterpolationModes.MAX)
    )

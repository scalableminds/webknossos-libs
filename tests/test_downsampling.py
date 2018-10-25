import logging
import numpy as np
from wkcuber.downsampling import (
    InterpolationModes,
    downsample_cube,
    downsample_cube_job,
    cube_addresses,
)
import wkw
from wkcuber.utils import WkwDatasetInfo, open_wkw
import shutil

WKW_CUBE_SIZE = 1024
CUBE_EDGE_LEN = 256

source_info = WkwDatasetInfo("testdata/WT1_wkw", "color", "uint8", 1)
target_info = WkwDatasetInfo("testoutput/WT1_wkw", "color", "uint8", 2)


def read_wkw(wkw_info, offset, size, block_type=None):
    with open_wkw(wkw_info, block_type=block_type) as wkw_dataset:
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
    addresses = cube_addresses(source_info)
    assert len(addresses) == 5 * 5 * 1

    assert min(addresses) == (0, 0, 0)
    assert max(addresses) == (4, 4, 0)


def downsample_test_helper(use_compress):
    try:
        shutil.rmtree(target_info.dataset_path)
    except:
        pass

    offset = (1, 2, 0)
    source_buffer = read_wkw(
        source_info,
        tuple(a * WKW_CUBE_SIZE * 2 for a in offset),
        (CUBE_EDGE_LEN * 2,) * 3,
    )
    assert np.any(source_buffer != 0)

    downsample_cube_job(
        source_info,
        target_info,
        2,
        InterpolationModes.MAX,
        CUBE_EDGE_LEN,
        offset,
        use_compress,
    )

    assert np.any(source_buffer != 0)
    block_type = (
        wkw.Header.BLOCK_TYPE_LZ4HC if use_compress else wkw.Header.BLOCK_TYPE_RAW
    )
    target_buffer = read_wkw(
        target_info,
        tuple(a * WKW_CUBE_SIZE for a in offset),
        (CUBE_EDGE_LEN,) * 3,
        block_type,
    )
    assert np.any(target_buffer != 0)

    assert np.all(
        target_buffer == downsample_cube(source_buffer, 2, InterpolationModes.MAX)
    )


def test_downsample_cube_job():
    downsample_test_helper(False)


def test_compressed_downsample_cube_job():
    downsample_test_helper(True)

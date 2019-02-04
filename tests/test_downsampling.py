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
from wkcuber.downsampling import _mode, non_linear_filter_3d
import shutil

WKW_CUBE_SIZE = 1024
CUBE_EDGE_LEN = 256

source_info = WkwDatasetInfo("testdata/WT1_wkw", "color", "uint8", 1)
target_info = WkwDatasetInfo("testoutput/WT1_wkw", "color", "uint8", 2)


def read_wkw(wkw_info, offset, size, **kwargs):
    with open_wkw(wkw_info, **kwargs) as wkw_dataset:
        return wkw_dataset.read(offset, size)


def test_downsample_cube():
    buffer = np.zeros((CUBE_EDGE_LEN,) * 3, dtype=np.uint8)
    buffer[:, :, :] = np.arange(0, CUBE_EDGE_LEN)

    output = downsample_cube(buffer, (2, 2, 2), InterpolationModes.MODE)

    assert output.shape == (CUBE_EDGE_LEN // 2,) * 3
    assert buffer[0, 0, 0] == 0
    assert buffer[0, 0, 1] == 1
    assert np.all(output[:, :, :] == np.arange(0, CUBE_EDGE_LEN, 2))


def test_downsample_mode():

    a = np.array([[1, 3, 4, 2, 2, 7], [5, 2, 2, 1, 4, 1], [3, 3, 2, 2, 1, 1]])

    result = _mode(a)
    expected_result = np.array([1, 3, 2, 2, 1, 1])

    assert np.all(result == expected_result)


def test_downsample_median():

    a = np.array([[1, 3, 4, 2, 2, 7], [5, 2, 2, 1, 4, 1], [3, 3, 2, 2, 1, 1]])

    result = np.median(a, axis=0)
    expected_result = np.array([3, 3, 2, 2, 2, 1])

    assert np.all(result == expected_result)


def test_non_linear_filter_reshape():
    a = np.array([[[1, 3], [1, 4]], [[4, 2], [3, 1]]], dtype=np.uint8)

    a_filtered = non_linear_filter_3d(a, [2, 2, 2], _mode)
    assert a_filtered.dtype == np.uint8
    expected_result = [1]
    assert np.all(expected_result == a_filtered)

    a = np.array([[[1, 3], [1, 4]], [[4, 3], [3, 1]]], np.uint32)

    a_filtered = non_linear_filter_3d(a, [2, 2, 1], _mode)
    assert a_filtered.dtype == np.uint32
    expected_result = [1, 3]
    assert np.all(expected_result == a_filtered)


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
    )[0]
    assert np.any(source_buffer != 0)

    downsample_cube_job(
        source_info,
        target_info,
        (2, 2, 2),
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
        block_type=block_type,
    )[0]
    assert np.any(target_buffer != 0)

    assert np.all(
        target_buffer
        == downsample_cube(source_buffer, (2, 2, 2), InterpolationModes.MAX)
    )


def test_downsample_cube_job():
    downsample_test_helper(False)


def test_compressed_downsample_cube_job():
    downsample_test_helper(True)


def test_downsample_multi_channel():
    source_info = WkwDatasetInfo("testoutput/multi-channel-test", "color", "uint8", 1)
    target_info = WkwDatasetInfo("testoutput/multi-channel-test", "color", "uint8", 2)
    try:
        shutil.rmtree(source_info.dataset_path)
        shutil.rmtree(target_info.dataset_path)
    except:
        pass

    offset = (0, 0, 0)
    num_channels = 3
    size = (32, 32, 10)
    source_data = (
        128 * np.random.randn(num_channels, size[0], size[1], size[2])
    ).astype("uint8")
    file_len = 32

    with open_wkw(
        source_info, num_channels=num_channels, file_len=file_len
    ) as wkw_dataset:
        print("writing source_data shape", source_data.shape)
        wkw_dataset.write(offset, source_data)
    assert np.any(source_data != 0)

    downsample_cube_job(
        source_info,
        target_info,
        (2, 2, 2),
        InterpolationModes.MAX,
        CUBE_EDGE_LEN,
        tuple(a * WKW_CUBE_SIZE for a in offset),
        False,
    )

    channels = []
    for channel_index in range(num_channels):
        channels.append(
            downsample_cube(
                source_data[channel_index], (2, 2, 2), InterpolationModes.MAX
            )
        )
    joined_buffer = np.stack(channels)

    target_buffer = read_wkw(
        target_info,
        tuple(a * WKW_CUBE_SIZE for a in offset),
        list(map(lambda x: x // 2, size)),
        file_len=file_len,
    )
    assert np.any(target_buffer != 0)

    assert np.all(target_buffer == joined_buffer)

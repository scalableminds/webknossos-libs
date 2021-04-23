from wkcuber.image_readers import ImageReaderManager
import numpy as np


def test_rgb_tiff_case() -> None:
    image_reader_manager = ImageReaderManager()
    result = image_reader_manager.read_array(
        "./testdata/rgb_tiff/test_rgb.tif", np.uint8, 0
    )
    assert result.shape == (32, 32, 3, 1)
    assert np.array_equal(result[0][0], [[0], [255], [0]])
    assert np.array_equal(result[0][20], [[255], [0], [0]])
    assert np.array_equal(result[30][0], [[0], [0], [255]])
    assert np.array_equal(result[16][16], [[255], [255], [255]])


def test_single_channel_conversion() -> None:
    image_reader_manager = ImageReaderManager()
    multi_channel_data = image_reader_manager.read_array(
        "./testdata/rgb_tiff/test_rgb.tif", np.uint8, 0
    )
    single_channel_data = np.empty((3, 32, 32, 1, 1), multi_channel_data.dtype)
    for i in range(3):
        single_channel_data[i] = image_reader_manager.read_array(
            "./testdata/rgb_tiff/test_rgb.tif", np.uint8, 0, i
        )

    single_channel_data = single_channel_data.transpose((1, 2, 0, 3, 4))
    single_channel_data = single_channel_data[:, :, :, 0]
    assert np.array_equal(multi_channel_data, single_channel_data)

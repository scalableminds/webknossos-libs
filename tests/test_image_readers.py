from wkcuber.image_readers import ImageReaderManager
import numpy as np


def test_rgb_tiff_case() -> None:
    image_reader_manager = ImageReaderManager()
    result = image_reader_manager.read_array(
        "./testdata/rgb_tiff/test_rgb.tif", np.uint8, 0
    )
    assert result.shape == (32, 32, 3, 1)
    assert np.all(result[0][0] == [[0], [255], [0]])
    assert np.all(result[0][20] == [[255], [0], [0]])
    assert np.all(result[30][0] == [[0], [0], [255]])
    assert np.all(result[16][16] == [[255], [255], [255]])

import numpy as np
from wkcuber.convert_nifti import to_target_datatype


def test_to_target_datatype_segmentation_layer():
    data = np.array([[[[0, 0, 1, 5, 27], [3200, 3, 2, 1, 5], [0, 3, 2, 1, 5]],
                      [[3200, 3, 2, 1, 5], [0, 0, 1, 5, 27], [6, 6, 6, 6, 6]]]]).astype("uint32")

    labelled_data = np.array([[[[0, 0, 1, 4, 6], [7, 3, 2, 1, 4], [0, 3, 2, 1, 4]],
                      [[7, 3, 2, 1, 4], [0, 0, 1, 4, 6], [5, 5, 5, 5, 5]]]])

    target_dtype = np.dtype("uint32")

    converted_segmentation = to_target_datatype(data, target_dtype, is_segmentation_layer=True)
    converted_no_segmentation = to_target_datatype(data, target_dtype, is_segmentation_layer=False)

    assert np.all(converted_segmentation == labelled_data)
    assert np.all(converted_no_segmentation == data)

import numpy as np
from wkcuber._internal.utils import pad_or_crop_to_size_and_topleft


def test_pad_or_crop_to_size_and_topleft() -> None:
    target_topleft = np.array((0, 50, 36, 1))
    target_size = np.array((1, 156, 112, 30))

    test_shapes = [(1, 117, 95, 16), (1, 128, 128, 16), (1, 196, 196, 12)]
    expected_voxel_counts = [117 * 95 * 16, 128 * 112 * 16, 156 * 112 * 12]

    for shape, expected_voxel_count in zip(test_shapes, expected_voxel_counts):
        cube_data = np.ones(shape)
        cube_data = pad_or_crop_to_size_and_topleft(
            cube_data, target_size, target_topleft
        )
        assert cube_data.shape[1:3] == tuple((target_size + target_topleft)[1:3])

        cube_data_from_top_left = cube_data[
            :, target_topleft[1] :, target_topleft[2] :, :
        ]
        cube_data_from_top_left_to_size = cube_data_from_top_left[
            :, : target_size[1], : target_size[2], :
        ]
        assert cube_data_from_top_left.shape[0:3] == tuple(target_size)[0:3]
        assert np.sum(cube_data) == expected_voxel_count
        assert np.sum(cube_data_from_top_left_to_size) == expected_voxel_count

import wkw
import sys
import numpy as np

path_a = sys.argv[1]
path_b = sys.argv[2]

with wkw.Dataset.open(path_a) as ds_a, wkw.Dataset.open(path_b) as ds_b:
    assert ds_a.header.version == ds_b.header.version
    assert ds_a.header.block_len == ds_b.header.block_len
    assert ds_a.header.file_len == ds_b.header.file_len
    assert ds_a.header.voxel_type == ds_b.header.voxel_type
    assert ds_a.header.num_channels == ds_b.header.num_channels
    data_a = ds_a.read((0, 0, 0), (1024, 1024, 1024))
    data_b = ds_b.read((0, 0, 0), (1024, 1024, 1024))
    assert np.all(data_a == data_b)

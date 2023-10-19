import shutil

import numpy as np
from cluster_tools import WrappedProcessPoolExecutor

import webknossos as wk


def main():
    inpath = "/home/f/Downloads/b1-c5-megafields-corrected/tif"
    outpath = "/home/f/Downloads/b1-c5-megafields-corrected/wkw_mag1"
    shutil.rmtree(outpath)

    out_ds = wk.Dataset(outpath, voxel_size=(1, 1, 1))

    layer = out_ds.add_layer_from_images(
        images=inpath,
        layer_name="color",
        executor=WrappedProcessPoolExecutor(max_workers=1),
        batch_size=32,
    )
    out_ds.downsample(
        executor=WrappedProcessPoolExecutor(max_workers=2), sampling_mode="constant_z"
    )
    out_ds.compress(executor=WrappedProcessPoolExecutor(max_workers=2))


if __name__ == "__main__":
    main()

import shutil

import numpy as np
from cluster_tools import WrappedProcessPoolExecutor

import webknossos as wk


def main():
    inpath = "/home/f/Downloads/bladder"
    outpath = "/home/f/Downloads/bladder-wkw"
    # shutil.rmtree(outpath)

    out_ds = wk.Dataset(outpath, voxel_size=(1, 1, 1))

    layer = out_ds.add_layer_from_images(
        images=inpath,
        layer_name="color",
        executor=WrappedProcessPoolExecutor(max_workers=2),
        use_bioformats=False,
    )


#    out_ds.downsample(executor=WrappedProcessPoolExecutor(max_workers=2))


if __name__ == "__main__":
    main()

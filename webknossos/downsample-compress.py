import shutil

import numpy as np
from cluster_tools import WrappedProcessPoolExecutor

import webknossos as wk


def main():
    ds = wk.Dataset(
        "/home/f/scm/code/webknossos/binaryData/sample_organization/l4_v2_sample"
    )
    ds.downsample(executor=WrappedProcessPoolExecutor(max_workers=2))
    ds.compress(executor=WrappedProcessPoolExecutor(max_workers=2))


if __name__ == "__main__":
    main()

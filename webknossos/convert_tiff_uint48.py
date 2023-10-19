from cluster_tools import WrappedProcessPoolExecutor

import webknossos as wk


def main():
    ds = wk.Dataset.from_images(
        input_path="/home/f/scm/datasets/H26_SP_DAPI_WGA_actinin_cadherin_xz_stack",
        output_path="/home/f/scm/datasets/H26_SP_DAPI_WGA_actinin_cadherin_xz_stack_wkw",
        voxel_size=(0.162, 0.162, 1),
        name="H26_SP_DAPI_WGA_actinin_cadherin_xz_stack",
        use_bioformats=False,
        executor=WrappedProcessPoolExecutor(max_workers=4),
    )
    # ds.compress(executor=WrappedProcessPoolExecutor(max_workers=4))
    # ds.downsample(sampling_mode="constant_z", executor=WrappedProcessPoolExecutor(max_workers=4))


if __name__ == "__main__":
    main()

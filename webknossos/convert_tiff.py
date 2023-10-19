from cluster_tools import WrappedProcessPoolExecutor

import webknossos as wk


def main():
    ds = wk.Dataset.from_images(
        input_path="/home/f/scm/datasets/sample_tiff",
        output_path="/home/f/scm/datasets/sample_tiff_wkw",
        voxel_size=(64, 64, 35),
        name="ZF-No2-retina-no-rough",
        use_bioformats=False,
        executor=WrappedProcessPoolExecutor(max_workers=4),
    )
    ds.compress(executor=WrappedProcessPoolExecutor(max_workers=4))
    ds.downsample(
        sampling_mode="constant_z", executor=WrappedProcessPoolExecutor(max_workers=4)
    )


if __name__ == "__main__":
    main()

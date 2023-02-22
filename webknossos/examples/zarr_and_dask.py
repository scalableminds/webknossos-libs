import dask.array as da

import webknossos as wk

MAG = wk.Mag("8-8-2")


def main() -> None:
    # Remote datasets are read-only, but can be used similar to normal datasets:
    l4_sample_dataset = wk.Dataset.open_remote(
        "https://webknossos.org/datasets/scalable_minds/l4_sample"
    )

    layer = l4_sample_dataset.get_layer("color")
    mag_view = layer.get_mag(MAG)

    zarr_array = mag_view.get_zarr_array()
    dask_array = da.from_array(zarr_array, chunks=(1, 256, 256, 256))[
        (0,) + mag_view.bounding_box.in_mag(MAG).to_slices()
    ]

    mean_value = dask_array.mean().compute()
    print("Mean:", mean_value)


if __name__ == "__main__":
    main()

import dask.array as da

import webknossos as wk


def main() -> None:
    # Remote datasets are read-only, but can be used similar to normal datasets:
    l4_sample_dataset = wk.Dataset.open_remote(
        "https://webknossos.org/datasets/scalable_minds/l4_sample"
    )

    layer = l4_sample_dataset.get_layer("color")
    mag = layer.get_finest_mag()

    zarr_array = mag.get_zarr_array()
    dask_array = da.from_array(zarr_array, chunks=(1, 256, 256, 256))[
        (0,) + layer.bounding_box.to_slices()
    ]

    mean_value = dask_array.mean().compute()
    print("Mean:", mean_value)


if __name__ == "__main__":
    main()

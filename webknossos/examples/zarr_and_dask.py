import chunk

import dask.array as da
import numpy as np

import webknossos as wk


def map_func(chunk, **_) -> np.ndarray:
    return np.array([chunk.min(), chunk.max(), chunk.sum(), chunk.size], dtype="uint64")


def combine_func(chunk_stats, **_) -> np.ndarray:
    chunk_stats = np.array(chunk_stats).reshape(-1, 4)
    return np.array(
        [
            chunk_stats[:, 0].min(),
            chunk_stats[:, 1].max(),
            chunk_stats[:, 2].sum(),
            chunk_stats[:, 3].sum(),
        ],
        dtype="uint64",
    )


def agg_func(chunk_stats, **_) -> np.ndarray:
    chunk_stats = np.array(chunk_stats).reshape(-1, 4)
    return np.array(
        [
            chunk_stats[:, 0].min(),
            chunk_stats[:, 1].max(),
            chunk_stats[:, 2].sum() / chunk_stats[:, 3].sum(),
        ],
        dtype="uint64",
    )


def main() -> None:
    # Remote datasets are read-only, but can be used similar to normal datasets:
    l4_sample_dataset = wk.Dataset.open_remote(
        "https://webknossos.org/datasets/scalable_minds/l4_sample"
    )

    layer = l4_sample_dataset.get_layer("color")
    mag = layer.get_finest_mag()

    zarr_array = mag.get_zarr_array()
    dask_array = da.from_array(zarr_array, chunks=(1, 256, 256, 256))[
        0, 3072:, 3072:, 512:
    ]

    min, max, mean = da.reduction(
        dask_array,
        chunk=map_func,
        combine=combine_func,
        aggregate=agg_func,
        dtype="uint64",
        concatenate=False,
    ).compute()

    print("Min:", min)
    print("Max:", max)
    print("Mean:", mean)


if __name__ == "__main__":
    main()

from pathlib import Path

import pims

import webknossos as wk

TIF_PATH = Path(".") / "webknossos" / "testdata" / "4D" / "4D-series.ome.tif"
ZARR_PATH = Path("..") / "idr0101A-4d-zarr"


def from_images_import():
    reader = pims.ImageSequenceND([str(TIF_PATH)])
    print(reader.sizes)
    print(isinstance(reader, pims.FramesSequenceND))
    # wk.Dataset.from_images(
    #     input_path, output_path, voxel_size=(10, 10, 10), data_format="zarr3"
    # )


def open_existing_dataset():
    ds = wk.Dataset.open(ZARR_PATH)
    color_layer = ds.get_color_layers()[0]
    finest_mag = color_layer.get_finest_mag()
    layers = ds.layers


def main() -> None:
    """Imports a dataset with more than 3 dimensions."""
    # from_images_import()
    open_existing_dataset()


if __name__ == "__main__":
    main()

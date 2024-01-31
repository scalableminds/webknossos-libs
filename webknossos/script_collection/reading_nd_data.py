from pathlib import Path

import pims
from tifffile import imwrite

import webknossos as wk
from webknossos.geometry.nd_bounding_box import NDBoundingBox

TIF_PATH = Path(".") / "webknossos" / "testdata" / "4D" / "4D_series"
OUTPUT = Path(".") / "testoutput"


def from_images_import():
    # reader = pims.Bioformats(str(TIF_PATH))
    # print(reader.sizes)
    # print(isinstance(reader, pims.FramesSequenceND))
    dataset = wk.Dataset.from_images(
        TIF_PATH,
        OUTPUT,
        voxel_size=(10, 10, 10),
        data_format="zarr3",
        use_bioformats=True,
    )
    layer = dataset.get_color_layers()[0]
    mag_view = layer.get_finest_mag()
    read_bbox = NDBoundingBox(topleft=(0, 0, 0, 0), size=(1, 1, 439, 167), axes=("t", "z", "y", "x"), index=(1,2,3,4))
    data = mag_view.read(absolute_bounding_box=read_bbox)
    assert data.shape == (1, ) + read_bbox.size
    data = mag_view.read(absolute_bounding_box=NDBoundingBox(topleft=(0, 0, 0, 0), size=(1, 1, 439, 167), axes=("t", "z", "y", "x"), index=(1,2,3,4)))
  
    imwrite("l4_sample_tiff/test.tiff", data)

    for bbox in layer.bounding_box.chunk((439, 167, 5)):
        with mag_view.get_buffered_slice_reader(absolute_bounding_box=bbox) as reader:
            for slice_data in reader:

                imwrite(
                    f"l4_sample_tiff/tiff_from_bbox{bbox}.tiff",
                    slice_data,
                )


def open_existing_dataset():
    ds = wk.Dataset.open(OUTPUT)
    # color_layer = ds.get_color_layers()[0]
    # finest_mag = color_layer.get_finest_mag()
    # layers = ds.layers
    # finest_mag.read()
    ds.copy_dataset("../copied_dataset")


def main() -> None:
    """Imports a dataset with more than 3 dimensions."""
    from_images_import()
    open_existing_dataset()


if __name__ == "__main__":
    main()

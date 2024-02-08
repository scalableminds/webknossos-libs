from pathlib import Path

import numpy as np
import tifffile

# import pims
from tifffile import imwrite

import webknossos as wk
from webknossos.geometry.nd_bounding_box import NDBoundingBox

TIF_PATH = Path(".") / "webknossos" / "testdata" / "4D" / "4D_series"
OUTPUT = Path(".") / "testdata"


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
    data = mag_view.read()[0, 0, 0, :, :]  # absolute_bounding_box=read_bbox)
    # assert data.shape == (1,)+read_bbox.size

    imwrite("l4_sample_tiff/test.tiff", data)

    for bbox in layer.bounding_box.chunk((439, 167, 5)):
        with mag_view.get_buffered_slice_reader(absolute_bounding_box=bbox) as reader:
            for slice_data in reader:
                imwrite(
                    f"l4_sample_tiff/tiff_from_bbox{bbox}.tiff",
                    slice_data,
                )


def test_compare_tifffile() -> None:
    ds = wk.Dataset(OUTPUT, (1, 1, 1))
    l = ds.add_layer_from_images(
        "./webknossos/testdata/tiff/test.02*.tiff",
        layer_name="compare_tifffile",
        category="segmentation",
        topleft=(100, 100, 55),
        chunk_shape=(8, 8, 8),
        chunks_per_shard=(8, 8, 8),
    )
    assert l.bounding_box.topleft == wk.Vec3Int(100, 100, 55)
    data = l.get_finest_mag().read()[0, :, :]
    for z_index in range(0, data.shape[-1]):
        with tifffile.TiffFile("webknossos/testdata/tiff/test.0200.tiff") as tif_file:
            comparison_slice = tif_file.asarray().T
        assert np.array_equal(data[:, :, z_index], comparison_slice)


def open_existing_dataset():
    ds = wk.Dataset.open(OUTPUT)
    layer = ds.get_color_layers()[0]
    mag_view = layer.get_finest_mag()
    read_bbox = NDBoundingBox(
        topleft=(0, 0, 0, 0),
        size=(1, 1, 439, 167),
        axes=("t", "z", "y", "x"),
        index=(1, 2, 3, 4),
    )
    data = mag_view.read(absolute_bounding_box=read_bbox)
    assert data.shape == (1,) + read_bbox.size
    data = mag_view.read(
        absolute_bounding_box=NDBoundingBox(
            topleft=(0, 0, 0, 0),
            size=(1, 1, 439, 167),
            axes=("t", "z", "y", "x"),
            index=(1, 2, 3, 4),
        )
    )

    imwrite("l4_sample_tiff/test.tiff", data)
    # color_layer = ds.get_color_layers()[0]
    # finest_mag = color_layer.get_finest_mag()
    # layers = ds.layers
    # finest_mag.read()
    ds.copy_dataset("../copied_dataset")


def main() -> None:
    """Imports a dataset with more than 3 dimensions."""
    from_images_import()
    # test_compare_tifffile()
    # open_existing_dataset()


if __name__ == "__main__":
    main()
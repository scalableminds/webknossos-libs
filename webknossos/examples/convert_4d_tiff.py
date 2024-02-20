from pathlib import Path

import webknossos as wk


def main() -> None:
    # Create a WEBKNOSSOS dataset from a 4D tiff image
    dataset = wk.Dataset.from_images(
        Path(__file__).parent.parent / "testdata" / "4D" / "4D_series",
        "testoutput/4D_series",
        voxel_size=(10, 10, 10),
        data_format="zarr3",
        use_bioformats=True,
    )

    # Access the first color layer and the Mag 1 view of this layer
    layer = dataset.get_color_layers()[0]
    mag_view = layer.get_finest_mag()

    # Read the data of the dataset within a bounding box
    read_bbox = wk.NDBoundingBox(
        topleft=(2, 0, 0, 0),
        size=(1, 5, 167, 439),
        axes=("t", "z", "y", "x"),
        index=(1, 2, 3, 4),
    )
    data = mag_view.read(absolute_bounding_box=read_bbox)
    # data.shape -> (1, 1, 5, 167, 439)

    # Write some data to a given position
    mag_view.write(data, absolute_bounding_box=read_bbox.offset((2, 0, 0, 0)))


if __name__ == "__main__":
    main()

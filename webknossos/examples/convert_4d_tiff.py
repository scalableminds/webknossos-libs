from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import httpx

import webknossos as wk

FOUR_D_SAMPLE_URL = "https://static.webknossos.org/data/wklibs-samples/4D.zip"


def main() -> None:
    # Download and extract the sample 4D tiff image
    source_dir = Path("testoutput/4D_source")
    source_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".zip") as archive_file:
        with httpx.stream("GET", FOUR_D_SAMPLE_URL, follow_redirects=True) as response:
            for chunk in response.iter_bytes():
                archive_file.write(chunk)
        with ZipFile(archive_file, "r") as zip_file:
            zip_file.extractall(source_dir)

    # Create a WEBKNOSSOS dataset from a 4D tiff image
    dataset = wk.Dataset.from_images(
        source_dir / "4D" / "4D_series",
        "testoutput/4D_series",
        layer_category=wk.COLOR_CATEGORY,
        data_format="zarr3",
        voxel_size=(10, 10, 10),
    )
    dataset.downsample()
    dataset.compress()

    # Access the first color layer and the Mag 1 view of this layer
    layer = dataset.get_color_layers()[0]
    mag_view = layer.get_finest_mag()

    # To get the bounding box of the dataset use layer.bounding_box
    # -> NDBoundingBox(topleft=(0, 0, 0, 0), size=(7, 5, 167, 439), axes=('t', 'z', 'y', 'x'))

    # Read all data of the dataset
    data = mag_view.read()
    # data.shape -> (1, 7, 5, 167, 439) # first value is the channel dimension

    # Read data for a specific time point (t=3) of the dataset
    data = mag_view.read(
        absolute_bounding_box=layer.bounding_box.with_bounds("t", 3, 1)
    )
    # data.shape -> (1, 1, 5, 167, 439)

    # Create a NDBoundingBox to read data from a specific region of the dataset
    read_bbox = wk.NDBoundingBox(
        topleft=(2, 0, 67, 39),
        size=(2, 5, 100, 400),
        axes=("t", "z", "y", "x"),
        index=(1, 2, 3, 4),
    )
    data = mag_view.read(absolute_bounding_box=read_bbox)
    # data.shape -> (1, 2, 5, 100, 400) # first value is the channel dimension

    # Write some data to a given position
    mag_view.write(
        data,
        absolute_bounding_box=read_bbox.offset((2, 0, 0, 0)),
        allow_resize=True,
        allow_unaligned=True,
    )


if __name__ == "__main__":
    main()

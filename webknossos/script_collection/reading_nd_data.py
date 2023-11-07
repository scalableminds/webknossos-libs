from pathlib import Path

import pims

import webknossos as wk


def main(input_path: Path, output_path: Path) -> None:
    """Imports a dataset with more than 3 dimensions."""

    wk.Dataset.from_images(
        input_path, output_path, voxel_size=(10, 10, 10), data_format="zarr3"
    )
    # reader = pims.Bioformats(input_path)
    # print(reader.sizes)
    # print(isinstance(reader, pims.FramesSequenceND))


if __name__ == "__main__":
    input_path = Path(".") / "webknossos" / "testdata" / "4D"  # / "4D-series.ome.tif"
    output_path = Path(".") / "webknossos" / "testoutput" / "4D"
    main(input_path, output_path)

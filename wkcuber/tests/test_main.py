from pathlib import Path
import numpy as np
import pytest
from wkcuber.utils import (
    setup_logging,
)
from webknossos import Dataset
from wkcuber.__main__ import create_parser, cube_with_args
from tifffile import TiffWriter

TESTOUTPUT_DIR = Path("testoutput")


@pytest.mark.parametrize("category", ["color", "segmentation"])
def test_main(category: str) -> None:
    input_folder = TESTOUTPUT_DIR / "raw_dataset" / category
    input_folder.mkdir(parents=True, exist_ok=True)

    raw_file = input_folder / "input.tif"

    input_dtype = "uint32"
    shape = 64, 128, 256
    data = np.arange(np.prod(shape), dtype=input_dtype).reshape(shape)
    with TiffWriter(raw_file) as tif:
        tif.write(data.transpose([2, 1, 0]))

    output_path = TESTOUTPUT_DIR / "output_2"
    output_path.mkdir()

    args_list = [
        str(TESTOUTPUT_DIR / "raw_dataset"),
        str(output_path),
        "--jobs",
        "1",
        "--scale",
        "11,11,11",
        "--max_mag",
        "4",
    ]

    args = create_parser().parse_args(args_list)
    cube_with_args(args)

    dataset = Dataset.open(output_path)
    if category == "color":
        layer = dataset.get_color_layers()[0]
    else:
        layer = dataset.get_segmentation_layers()[0]
    mag_view = layer.get_mag(1)
    view = mag_view.get_view()
    read_data = view.read()

    assert view.size == shape
    assert view.get_dtype() == data.dtype
    assert np.array_equal(
        read_data[0],
        data,
    )


if __name__ == "__main__":
    from argparse import Namespace

    setup_logging(Namespace(verbose=False))
    test_main("color")

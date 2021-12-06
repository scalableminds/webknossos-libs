from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pytest

from webknossos import Dataset
from wkcuber.convert_raw import create_parser, main


TESTOUTPUT_DIR = Path("testoutput")


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("flip_axes", [None, (1, 2)])
def test_main(order: str, flip_axes: Optional[Tuple[int, int]]) -> None:
    raw_file = TESTOUTPUT_DIR / "input.raw"

    input_dtype = "float32"
    shape = 64, 128, 256
    data = np.arange(np.prod(shape), dtype=input_dtype).reshape(shape, order=order)
    with raw_file.open("wb") as f:
        f.write(data.tobytes(order=order))

    output_path = TESTOUTPUT_DIR / "output"
    output_path.mkdir()

    args_list = [
        str(raw_file),
        str(output_path),
        "--input_dtype",
        input_dtype,
        "--shape",
        ",".join(str(i) for i in shape),
        "--order",
        order,
        "--jobs",
        "1",
    ]
    if flip_axes is not None:
        args_list.extend(["--flip_axes", ",".join(str(a + 1) for a in flip_axes)])

    args = create_parser().parse_args(args_list)
    main(args)

    dataset = Dataset(output_path)
    layer = dataset.get_color_layer()
    mag_view = layer.get_mag(1)
    view = mag_view.get_view()
    read_data = view.read()

    assert view.size == shape
    assert view.get_dtype() == data.dtype
    assert np.array_equal(
        read_data[0],
        data if flip_axes is None else np.flip(data, flip_axes),
    )

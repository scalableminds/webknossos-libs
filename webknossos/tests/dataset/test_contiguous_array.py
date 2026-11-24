import io
from itertools import product
from typing import Literal

import numpy as np
import pytest

from webknossos.dataset._utils.contiguous_array import read_box

_SHAPE = (3, 4, 5)
_BOXES = [
    (slice(None), slice(None), slice(None)),
    (slice(1, 2), slice(None), slice(None)),  # whole planes: one read
    (slice(None), slice(1, 3), slice(None)),  # whole rows: one read per plane
    (slice(0, 3), slice(1, 3), slice(2, 4)),  # one read per row
    (slice(2, 9), slice(3, 99), slice(4, 5)),  # clipped like numpy
    (slice(1, 1), slice(None), slice(None)),  # empty
]


@pytest.mark.parametrize(
    "order, dtype, box", list(product("CF", ["uint8", ">u2", "<f8"], _BOXES))
)
def test_read_box_matches_numpy(
    order: Literal["C", "F"], dtype: str, box: tuple[slice, ...]
) -> None:
    array = np.arange(np.prod(_SHAPE)).astype(dtype).reshape(_SHAPE, order=order)
    offset = 7
    file = io.BytesIO(b"x" * offset + array.tobytes(order=order))

    result = read_box(
        file, offset=offset, shape=_SHAPE, dtype=dtype, order=order, box=box
    )

    assert result.dtype == array.dtype
    np.testing.assert_array_equal(result, array[box])


def test_read_box_raises_on_truncated_file() -> None:
    with pytest.raises(EOFError):
        read_box(
            io.BytesIO(b"\0" * 10),
            offset=0,
            shape=(4, 4),
            dtype="uint8",
            order="C",
            box=(slice(None), slice(None)),
        )


def test_read_box_rejects_steps() -> None:
    with pytest.raises(ValueError, match="step-1"):
        read_box(
            io.BytesIO(b"\0" * 16),
            offset=0,
            shape=(4, 4),
            dtype="uint8",
            order="C",
            box=(slice(None, None, 2), slice(None)),
        )

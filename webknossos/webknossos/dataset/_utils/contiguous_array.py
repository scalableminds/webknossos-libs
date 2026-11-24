"""Reads boxes out of an uncompressed array stored contiguously in a file,
with plain seeks and reads, so that any file object works, including remote
ones."""

import math
from collections.abc import Sequence
from itertools import product
from typing import IO, Literal

import numpy as np
from numpy.typing import DTypeLike


def read_box(
    file: IO[bytes],
    *,
    offset: int,
    shape: Sequence[int],
    dtype: DTypeLike,
    order: Literal["C", "F"],
    box: Sequence[slice],
) -> np.ndarray:
    """Reads `box` (one step-1 slice per axis, clipped to `shape` like numpy
    indexing) from the array of `shape` and `dtype` that is stored in `order`
    starting at byte `offset` of `file`.

    Issues one seek and read per contiguous run of bytes: the innermost axes
    the box spans completely are read together with the next outer axis.
    """
    if order == "F":
        # An F-order array is the C-order array with all axes reversed.
        return read_box(
            file,
            offset=offset,
            shape=shape[::-1],
            dtype=dtype,
            order="C",
            box=box[::-1],
        ).T

    dtype = np.dtype(dtype)
    ranges = [range(*s.indices(size)) for s, size in zip(box, shape)]
    if any(r.step != 1 for r in ranges):
        raise ValueError(f"Only step-1 slices are supported, got {box}.")

    result = np.empty([len(r) for r in ranges], dtype=dtype)
    if result.size == 0:
        return result

    strides = [
        dtype.itemsize * math.prod(shape[axis + 1 :]) for axis in range(len(shape))
    ]
    run_axis = len(shape) - 1
    while run_axis > 0 and len(ranges[run_axis]) == shape[run_axis]:
        run_axis -= 1
    run_offset = offset + ranges[run_axis].start * strides[run_axis]

    runs = result.reshape(math.prod(len(r) for r in ranges[:run_axis]), -1)
    for run, index in zip(runs, product(*ranges[:run_axis])):
        file.seek(run_offset + sum(i * stride for i, stride in zip(index, strides)))
        run_bytes = run.view(np.uint8)
        if file.readinto(run_bytes) != run_bytes.size:  # type: ignore[attr-defined]
            raise EOFError(
                f"The file ends before the {shape} array of {dtype} it should hold."
            )
    return result

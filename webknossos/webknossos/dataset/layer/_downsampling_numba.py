"""Numba implementations of the non-linear downsampling filters.

This module is only imported when the optional `numba` dependency is installed;
`_downsampling_utils` falls back to numpy implementations otherwise.
"""

from collections.abc import Callable

import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def _mode(input_array: np.ndarray) -> np.ndarray:
    values = np.zeros(input_array.shape[0], dtype=input_array.dtype)
    counter = np.zeros(input_array.shape[0], dtype=np.uint8)
    output_array = np.zeros(input_array.shape[1], dtype=input_array.dtype)
    for row_index in range(input_array.shape[1]):
        values[0] = input_array[0, row_index]
        counter[:] = 1
        value_offset = 1
        for col_index in range(1, input_array.shape[0]):
            value = input_array[col_index, row_index]
            found_value = False
            for i in range(
                value_offset
            ):  # Only iterate the values that were already seen
                if value == values[i]:
                    counter[i] = counter[i] + 1
                    found_value = True
                    break
            if not found_value:
                values[value_offset] = value
                value_offset += 1
        mode = values[np.argmax(counter)]
        output_array[row_index] = mode

    return output_array


# These kernels read the source buffer directly, which avoids the three full copies
# that reshaping it into an (elements-per-block, num-blocks) array costs.
# Within a block the elements are visited in the order `dy + fy * dx + fx * fy * dz`,
# which is the order `non_linear_filter_3d` produces and which determines how the
# mode filter breaks ties.


@numba.jit(nopython=True, nogil=True)
def _median_kernel(
    source: np.ndarray, target: np.ndarray, fx: int, fy: int, fz: int
) -> None:
    n = fx * fy * fz
    half = n // 2
    block = np.empty(n, dtype=source.dtype)
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                i = 0
                for dz in range(fz):
                    for dy in range(fy):
                        for dx in range(fx):
                            block[i] = source[x * fx + dx, y * fy + dy, z * fz + dz]
                            i += 1
                # Insertion sort, n is tiny (8 for an isotropic step).
                for a in range(1, n):
                    value = block[a]
                    b = a - 1
                    while b >= 0 and block[b] > value:
                        block[b + 1] = block[b]
                        b -= 1
                    block[b + 1] = value
                if n % 2 == 1:
                    target[x, y, z] = block[half]
                else:
                    # np.median averages the two middle elements in float64 and
                    # truncates to the input dtype.
                    target[x, y, z] = (
                        np.float64(block[half - 1]) + np.float64(block[half])
                    ) / 2.0


@numba.jit(nopython=True, nogil=True)
def _median_kernel_2x2x2(source: np.ndarray, target: np.ndarray) -> None:
    # Batcher odd-even merge network for 8 elements, reduced to the comparators
    # needed to place the 4th and 5th smallest element.
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                a0 = source[2 * x, 2 * y, 2 * z]
                a1 = source[2 * x + 1, 2 * y, 2 * z]
                a2 = source[2 * x, 2 * y + 1, 2 * z]
                a3 = source[2 * x + 1, 2 * y + 1, 2 * z]
                a4 = source[2 * x, 2 * y, 2 * z + 1]
                a5 = source[2 * x + 1, 2 * y, 2 * z + 1]
                a6 = source[2 * x, 2 * y + 1, 2 * z + 1]
                a7 = source[2 * x + 1, 2 * y + 1, 2 * z + 1]

                b0 = min(a0, a1)
                b1 = max(a0, a1)
                b2 = min(a2, a3)
                b3 = max(a2, a3)
                b4 = min(a4, a5)
                b5 = max(a4, a5)
                b6 = min(a6, a7)
                b7 = max(a6, a7)

                c0 = min(b0, b2)
                c1 = min(b1, b3)
                c2 = max(b0, b2)
                c3 = max(b1, b3)
                c4 = min(b4, b6)
                c5 = min(b5, b7)
                c6 = max(b4, b6)
                c7 = max(b5, b7)

                d1 = min(c1, c2)
                d2 = max(c1, c2)
                d5 = min(c5, c6)
                d6 = max(c5, c6)

                e2 = min(d2, d6)
                e3 = min(c3, c7)
                e4 = max(c0, c4)
                e5 = max(d1, d5)

                f3 = min(e3, e5)
                f4 = max(e2, e4)

                target[x, y, z] = (
                    np.float64(min(f3, f4)) + np.float64(max(f3, f4))
                ) / 2.0


@numba.jit(nopython=True, nogil=True)
def _mode_kernel(
    source: np.ndarray, target: np.ndarray, fx: int, fy: int, fz: int
) -> None:
    n = fx * fy * fz
    values = np.empty(n, dtype=source.dtype)
    counts = np.empty(n, dtype=np.int64)
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                num_values = 0
                for dz in range(fz):
                    for dx in range(fx):
                        for dy in range(fy):
                            value = source[x * fx + dx, y * fy + dy, z * fz + dz]
                            found_value = False
                            for i in range(num_values):
                                if values[i] == value:
                                    counts[i] += 1
                                    found_value = True
                                    break
                            if not found_value:
                                values[num_values] = value
                                counts[num_values] = 1
                                num_values += 1
                # Ties are won by the value that occurs first in the block.
                best = 0
                for i in range(1, num_values):
                    if counts[i] > counts[best]:
                        best = i
                target[x, y, z] = values[best]


@numba.jit(nopython=True, nogil=True)
def _mode_kernel_2x2x2(source: np.ndarray, target: np.ndarray) -> None:
    # Each element counts only the matches that follow it, so the first occurrence
    # of a value carries its full count and ties are won by the earliest element.
    for z in range(target.shape[2]):
        for y in range(target.shape[1]):
            for x in range(target.shape[0]):
                a0 = source[2 * x, 2 * y, 2 * z]
                a1 = source[2 * x, 2 * y + 1, 2 * z]
                a2 = source[2 * x + 1, 2 * y, 2 * z]
                a3 = source[2 * x + 1, 2 * y + 1, 2 * z]
                a4 = source[2 * x, 2 * y, 2 * z + 1]
                a5 = source[2 * x, 2 * y + 1, 2 * z + 1]
                a6 = source[2 * x + 1, 2 * y, 2 * z + 1]
                a7 = source[2 * x + 1, 2 * y + 1, 2 * z + 1]

                c0 = (
                    1
                    + (a0 == a1)
                    + (a0 == a2)
                    + (a0 == a3)
                    + (a0 == a4)
                    + (a0 == a5)
                    + (a0 == a6)
                    + (a0 == a7)
                )
                c1 = (
                    1
                    + (a1 == a2)
                    + (a1 == a3)
                    + (a1 == a4)
                    + (a1 == a5)
                    + (a1 == a6)
                    + (a1 == a7)
                )
                c2 = 1 + (a2 == a3) + (a2 == a4) + (a2 == a5) + (a2 == a6) + (a2 == a7)
                c3 = 1 + (a3 == a4) + (a3 == a5) + (a3 == a6) + (a3 == a7)
                c4 = 1 + (a4 == a5) + (a4 == a6) + (a4 == a7)
                c5 = 1 + (a5 == a6) + (a5 == a7)
                c6 = 1 + (a6 == a7)

                best = a0
                best_count = c0
                if c1 > best_count:
                    best = a1
                    best_count = c1
                if c2 > best_count:
                    best = a2
                    best_count = c2
                if c3 > best_count:
                    best = a3
                    best_count = c3
                if c4 > best_count:
                    best = a4
                    best_count = c4
                if c5 > best_count:
                    best = a5
                    best_count = c5
                if c6 > best_count:
                    best = a6
                target[x, y, z] = best


def _can_use_kernel(data: np.ndarray, factors: list[int]) -> bool:
    # Floating point data uses the numpy implementation, because np.median
    # accumulates in the input dtype for float32 and has its own NaN semantics.
    return data.ndim == 3 and len(factors) == 3 and data.dtype.kind in "iu"


def _apply_kernel(
    data: np.ndarray,
    factors: list[int],
    kernel: Callable[[np.ndarray, np.ndarray, int, int, int], None],
    kernel_2x2x2: Callable[[np.ndarray, np.ndarray], None],
) -> np.ndarray:
    assert not any(d % factor > 0 for (d, factor) in zip(data.shape, factors))
    fx, fy, fz = factors
    target = np.empty(
        (data.shape[0] // fx, data.shape[1] // fy, data.shape[2] // fz),
        dtype=data.dtype,
        order="F",
    )
    if fx == 2 and fy == 2 and fz == 2:
        kernel_2x2x2(data, target)
    else:
        kernel(data, target, fx, fy, fz)
    return target


def median_downsample(data: np.ndarray, factors: list[int]) -> np.ndarray:
    return _apply_kernel(data, factors, _median_kernel, _median_kernel_2x2x2)


def mode_downsample(data: np.ndarray, factors: list[int]) -> np.ndarray:
    return _apply_kernel(data, factors, _mode_kernel, _mode_kernel_2x2x2)

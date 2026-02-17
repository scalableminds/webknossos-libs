import operator
import re

import numpy as np
from numpy.typing import DTypeLike

properties_floating_type_to_python_type: dict[DTypeLike, np.dtype] = {
    "float": np.dtype("float32"),
    #  np.float: np.dtype("float32"),  # np.float is an alias for float
    float: np.dtype("float32"),
    "double": np.dtype("float64"),
}
python_floating_type_to_properties_type = {
    "float32": "float",
    "float64": "double",
}


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _convert_dtypes(
    dtype: DTypeLike,
    num_channels: int,
    *,
    dtype_per_layer_to_dtype_per_channel: bool,
) -> str:
    op = operator.truediv if dtype_per_layer_to_dtype_per_channel else operator.mul

    # split the dtype into the actual type and the number of bits
    # example: "uint24" -> ["uint", "24"]
    dtype_parts = re.split(r"(\d+)", str(dtype))
    # calculate number of bits for dtype_per_channel
    converted_dtype_parts = [
        (str(int(op(int(part), num_channels))) if _is_int(part) else part)
        for part in dtype_parts
    ]
    return "".join(converted_dtype_parts)


def _dtype_per_layer_to_dtype_per_channel(
    dtype_per_layer: DTypeLike, num_channels: int
) -> np.dtype:
    try:
        return np.dtype(
            _convert_dtypes(
                dtype_per_layer, num_channels, dtype_per_layer_to_dtype_per_channel=True
            )
        )
    except TypeError as e:
        raise TypeError(
            f"Converting dtype_per_layer to dtype_per_channel failed. Double check if the dtype_per_layer value is correct. Got {dtype_per_layer} and {num_channels} channels."
        ) from e


def _dtype_per_channel_to_dtype_per_layer(
    dtype_per_channel: DTypeLike, num_channels: int
) -> str:
    return _convert_dtypes(
        np.dtype(dtype_per_channel),
        num_channels,
        dtype_per_layer_to_dtype_per_channel=False,
    )


def element_class_to_dtype_per_channel(
    element_class: str, num_channels: int
) -> np.dtype:
    dtype_per_layer = properties_floating_type_to_python_type.get(
        element_class, element_class
    )
    return _dtype_per_layer_to_dtype_per_channel(dtype_per_layer, num_channels)


def dtype_per_channel_to_element_class(
    dtype_per_channel: DTypeLike, num_channels: int
) -> str:
    dtype_per_layer = _dtype_per_channel_to_dtype_per_layer(
        dtype_per_channel, num_channels
    )
    return python_floating_type_to_properties_type.get(dtype_per_layer, dtype_per_layer)

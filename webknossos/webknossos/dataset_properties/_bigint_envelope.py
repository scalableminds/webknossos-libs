"""Codec for WEBKNOSSOS’s “bigint envelope” JSON format.

Segment ids can use the full uint64 range, while JavaScript number can only
handle 2**53 - 1. WEBKNOSSOS therefore wraps such ids in an envelope in JSON:

    {"customJsonEncoding": "bigint", "value": "18446744073709551615"}

Plain JSON numbers are still accepted (and, for values that fit safely,
emitted)

Python’s `int` is arbitrary-precision, so the decoded value needs no special
handling on the Python side. Only the envelope itself needs to be
recognized and unwrapped/wrapped.
"""

from typing import Any

_ENCODING_KEY = "customJsonEncoding"
_ENCODING_VALUE = "bigint"

# The largest integer JavaScript can represent exactly. Values up to this are
# still written as plain JSON number.
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def structure_int_or_bigint_envelope(value: Any, _type: Any = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict) and value.get(_ENCODING_KEY) == _ENCODING_VALUE:
        return int(value["value"])
    return int(value)


def unstructure_int_maybe_as_bigint_envelope(value: int | None) -> Any:
    if value is None:
        return None
    if value > _JS_MAX_SAFE_INTEGER:
        return {_ENCODING_KEY: _ENCODING_VALUE, "value": str(value)}
    return value

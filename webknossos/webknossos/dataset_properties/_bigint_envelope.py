"""Codec for WEBKNOSSOS's "bigint envelope" JSON format.

Segment/agglomerate ids can use the full uint64 range, which JavaScript's
`number` type (and thus plain JSON numbers, as consumed by JS-based tools)
cannot represent exactly beyond 2**53 - 1. Starting with API version 15,
WEBKNOSSOS therefore wraps such ids in a self-describing envelope instead of
emitting them as a plain JSON number, e.g.:

    {"customJsonEncoding": "bigint", "value": "18446744073709551615"}

Plain JSON numbers are still accepted (and, for values that fit safely,
emitted) for backwards compatibility with older API versions and with
datasource-properties.json files written before this format was introduced.

Python's `int` is arbitrary-precision, so the decoded value needs no special
handling on the Python side - only the envelope itself needs to be
recognized and unwrapped/wrapped.
"""

from typing import Any

_ENCODING_KEY = "customJsonEncoding"
_ENCODING_VALUE = "bigint"

# The largest integer JavaScript can represent exactly. Values up to this are
# still written as a plain JSON number for backwards compatibility.
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def structure_int_or_bigint_envelope(value: Any, _type: Any = None) -> int | None:
    """Parses an int that may be encoded as a plain JSON number or as the
    bigint envelope shown above."""
    if value is None:
        return None
    if isinstance(value, dict) and value.get(_ENCODING_KEY) == _ENCODING_VALUE:
        return int(value["value"])
    return int(value)


def unstructure_int_maybe_as_bigint_envelope(value: int | None) -> Any:
    """Inverse of `structure_int_or_bigint_envelope`: keeps values that fit
    into a JS-safe integer as plain ints, and wraps larger values in the
    bigint envelope to avoid losing precision."""
    if value is None:
        return None
    if value > _JS_MAX_SAFE_INTEGER:
        return {_ENCODING_KEY: _ENCODING_VALUE, "value": str(value)}
    return value

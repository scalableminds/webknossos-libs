from typing import TypeVar

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]


def filter_none_values(_dict: dict[str, str | None]) -> dict[str, str]:
    """XML values must not be None."""
    return {key: value for key, value in _dict.items() if value is not None}


T = TypeVar("T")


def enforce_not_null(val: T | None) -> T:
    if val is None:
        raise ValueError("Value is None")
    return val


def as_int_unless_none(val: str | None) -> int | None:
    if val is None:
        return None
    return int(val)


def as_float_unless_none(val: str | None) -> float | None:
    if val is None:
        return None
    return float(val)

from typing import Dict, Optional, Tuple, TypeVar

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]


def filter_none_values(_dict: Dict[str, Optional[str]]) -> Dict[str, str]:
    """XML values must not be None."""
    return {key: value for key, value in _dict.items() if value is not None}


T = TypeVar("T")


def enforce_not_null(val: Optional[T]) -> T:
    if val is None:
        raise ValueError("Value is None")
    return val


def as_int_unless_none(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    return int(val)

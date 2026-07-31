import numpy as np
import pytest

from webknossos.geometry import Vec3Int, parse_vec3_float


def test_parse_vec3_float() -> None:
    assert parse_vec3_float((1.5, 2, 3)) == (1.5, 2.0, 3.0)
    assert parse_vec3_float([1.5, 2, 3]) == (1.5, 2.0, 3.0)
    assert parse_vec3_float(np.array([1.5, 2.0, 3.0])) == (1.5, 2.0, 3.0)
    assert parse_vec3_float(Vec3Int(1, 2, 3)) == (1.0, 2.0, 3.0)
    assert parse_vec3_float(iter([1.5, 2, 3])) == (1.5, 2.0, 3.0)

    # Integers are widened to floats, the result is always a plain tuple of floats
    result = parse_vec3_float((1, 2, 3))
    assert result == (1.0, 2.0, 3.0)
    assert isinstance(result, tuple)
    assert all(isinstance(value, float) for value in result)


def test_parse_vec3_float_rejects_invalid_values() -> None:
    for invalid in [(1, 2), (1, 2, 3, 4), [], np.zeros((3, 1)), np.zeros((2, 3))]:
        with pytest.raises(ValueError, match="three floats"):
            parse_vec3_float(invalid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="three floats"):
        parse_vec3_float(("a", "b", "c"))  # type: ignore[arg-type]

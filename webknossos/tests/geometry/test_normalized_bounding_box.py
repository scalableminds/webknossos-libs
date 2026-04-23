import pytest

from webknossos.geometry import (
    BoundingBox,
    NDBoundingBox,
    NormalizedBoundingBox,
    VecInt,
)
from webknossos.geometry.normalized_bounding_box import _DEFAULT_AXIS_ORDER


def test_from_wkw_dict_simple() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {"topLeft": [1, 2, 3], "width": 4, "height": 5, "depth": 6}
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.axes == ("c", "x", "y", "z")
    assert result.topleft_xyz.to_list() == [1, 2, 3]
    assert result.size_xyz.to_list() == [4, 5, 6]
    assert result.size.c == 1


def test_from_wkw_dict_with_axis_order() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [1, 2, 3],
            "width": 4,
            "height": 5,
            "depth": 6,
            "axisOrder": _DEFAULT_AXIS_ORDER,
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.axes == ("c", "x", "y", "z")
    assert result.topleft_xyz.to_list() == [1, 2, 3]
    assert result.size_xyz.to_list() == [4, 5, 6]
    assert result.size.c == 1


def test_from_wkw_dict_with_axis_order_implicit_c() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [1, 2, 3],
            "width": 4,
            "height": 5,
            "depth": 6,
            "axisOrder": {"x": 1, "y": 2, "z": 3},
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.axes == ("c", "x", "y", "z")
    assert result.topleft_xyz.to_list() == [1, 2, 3]
    assert result.size_xyz.to_list() == [4, 5, 6]
    assert result.size.c == 1


def test_from_wkw_dict_with_axis_order_missing_c() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [1, 2, 3],
            "width": 4,
            "height": 5,
            "depth": 6,
            "axisOrder": {"x": 2, "y": 1, "z": 0},
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.axes == ("z", "y", "x")
    assert result.topleft_xyz.to_list() == [1, 2, 3]
    assert result.size_xyz.to_list() == [4, 5, 6]
    with pytest.raises(KeyError):
        assert result.size.c == 0


def test_from_wkw_dict_with_num_channels() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [0, 0, 0],
            "width": 10,
            "height": 20,
            "depth": 30,
            "numChannels": 3,
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.size.c == 3
    assert result.size_xyz.to_list() == [10, 20, 30]


def test_from_wkw_dict_with_channel_index() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [0, 0, 0],
            "width": 10,
            "height": 10,
            "depth": 10,
            "channelIndex": 2,
            "numChannels": 1,
            "axisOrder": _DEFAULT_AXIS_ORDER,
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert result.topleft.c == 2


def test_from_wkw_dict_with_additional_axes() -> None:
    result = NormalizedBoundingBox.from_wkw_dict(
        {
            "topLeft": [0, 0, 0],
            "width": 10,
            "height": 10,
            "depth": 10,
            "additionalAxes": [{"name": "t", "bounds": [0, 5], "index": 4}],
            "axisOrder": {"c": 0, "x": 1, "y": 2, "z": 3, "t": 4},
            "numChannels": 1,
        }
    )
    assert isinstance(result, NormalizedBoundingBox)
    assert "t" in result.axes
    assert result.size[result.axes.index("t")] == 5
    assert result.topleft[result.axes.index("t")] == 0


def test_from_wkw_dict_additional_axes_requires_axis_order() -> None:
    with pytest.raises(AssertionError):
        NormalizedBoundingBox.from_wkw_dict(
            {
                "topLeft": [0, 0, 0],
                "width": 10,
                "height": 10,
                "depth": 10,
                "additionalAxes": [{"name": "t", "bounds": [0, 5], "index": 4}],
            }
        )


def test_to_wkw_dict_simple() -> None:
    bbox = BoundingBox((1, 2, 3), (4, 5, 6)).normalize_axes(1)
    result = bbox.to_wkw_dict()
    assert result == {
        "topLeft": [1, 2, 3],
        "width": 4,
        "height": 5,
        "depth": 6,
        "numChannels": 1,
        "axisOrder": _DEFAULT_AXIS_ORDER,
    }


def test_to_wkw_dict_with_num_channels() -> None:
    bbox = BoundingBox((0, 0, 0), (10, 20, 30)).normalize_axes(3)
    result = bbox.to_wkw_dict()
    assert result == {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 20,
        "depth": 30,
        "numChannels": 3,
        "axisOrder": _DEFAULT_AXIS_ORDER,
    }


def test_to_wkw_dict_with_nondefault_axis_order() -> None:
    bbox = NormalizedBoundingBox(
        (0, 0, 0, 0),
        (10, 10, 10, 3),
        axes="zyxc",
    )
    result = bbox.to_wkw_dict()
    assert result == {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 10,
        "depth": 10,
        "axisOrder": {"c": 3, "z": 0, "y": 1, "x": 2},
        "numChannels": 3,
    }


def test_to_wkw_dict_with_additional_axes() -> None:
    bbox = NormalizedBoundingBox(
        VecInt(c=0, x=0, y=0, z=0, t=2),
        VecInt(c=1, x=10, y=10, z=10, t=5),
        axes="cxyzt",
    )
    result = bbox.to_wkw_dict()
    assert result == {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 10,
        "depth": 10,
        "additionalAxes": [{"name": "t", "bounds": [2, 7], "index": 4}],
        "axisOrder": _DEFAULT_AXIS_ORDER,
        "numChannels": 1,
    }


def test_to_wkw_dict_with_channel_index() -> None:
    # channelIndex is only included in to_wkw_dict when axes are non-standard
    # (i.e., when additionalAxes are present, triggering the else branch)
    bbox = NormalizedBoundingBox(
        VecInt(c=2, x=0, y=0, z=0),
        VecInt(c=4, x=10, y=10, z=10),
        axes="cxyz",
    )
    result = bbox.to_wkw_dict()
    assert result["channelIndex"] == 2
    assert result["numChannels"] == 4


def test_round_trip_simple() -> None:
    d = {"topLeft": [1, 2, 3], "width": 4, "height": 5, "depth": 6}
    bbox = NormalizedBoundingBox.from_wkw_dict(d).normalize_axes(1)
    result = bbox.to_wkw_dict()
    assert result == {
        **d,
        "numChannels": 1,
        "axisOrder": _DEFAULT_AXIS_ORDER,
    }


def test_round_trip_with_num_channels() -> None:
    d = {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 20,
        "depth": 30,
        "numChannels": 3,
    }
    bbox = NormalizedBoundingBox.from_wkw_dict(d).normalize_axes(3)
    assert bbox.to_wkw_dict() == {**d, "axisOrder": _DEFAULT_AXIS_ORDER}


def test_round_trip_with_channel_index() -> None:
    d = {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 20,
        "depth": 30,
        "numChannels": 3,
        "channelIndex": 2,
    }
    bbox = NormalizedBoundingBox.from_wkw_dict(d).normalize_axes(3)
    assert bbox.to_wkw_dict() == {**d, "axisOrder": _DEFAULT_AXIS_ORDER}


def test_round_trip_with_additional_axes() -> None:
    d = {
        "topLeft": [0, 0, 0],
        "width": 10,
        "height": 10,
        "depth": 10,
        "additionalAxes": [{"name": "t", "bounds": [0, 5], "index": 4}],
        "axisOrder": _DEFAULT_AXIS_ORDER,
        "numChannels": 1,
    }
    bbox = NormalizedBoundingBox.from_wkw_dict(d).normalize_axes(1)
    assert NormalizedBoundingBox.from_wkw_dict(bbox.to_wkw_dict()) == bbox


def test_from_wkw_dict_returns_nd_bounding_box_type() -> None:
    """from_wkw_dict is defined on NormalizedBoundingBox but returns NDBoundingBox subclass."""
    result = NormalizedBoundingBox.from_wkw_dict(
        {"topLeft": [0, 0, 0], "width": 1, "height": 1, "depth": 1}
    )
    assert isinstance(result, NDBoundingBox)
    assert isinstance(result, NormalizedBoundingBox)

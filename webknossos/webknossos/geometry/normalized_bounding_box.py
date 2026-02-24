from collections.abc import Sequence

import attr

from .nd_bounding_box import NDBoundingBox
from .vec_int import VecInt

_DEFAULT_AXIS_ORDER = {"c": 0, "x": 1, "y": 2, "z": 3}


@attr.frozen
class Axis:
    name: str
    min: int
    size: int
    index: int


def _find_index_by_name(axes: Sequence[Axis], name: str) -> int:
    for i, axis in enumerate(axes):
        if axis.name == name:
            return i
    raise ValueError(f"Axis {name} not found in {axes}")


class NormalizedBoundingBox(NDBoundingBox):
    def denormalize(self) -> NDBoundingBox:
        if self.axes == ("c", "x", "y", "z"):
            from .bounding_box import BoundingBox

            return BoundingBox(
                topleft=self.topleft_xyz,
                size=self.size_xyz,
                name=self.name,
                color=self.color,
                is_visible=self.is_visible,
            )
        else:
            return NDBoundingBox(
                topleft=self.topleft,
                size=self.size,
                axes=self.axes,
                index=self.index,
                name=self.name,
                color=self.color,
                is_visible=self.is_visible,
            )

    def __eq__(self, other: object) -> bool:
        """Check equality with another bounding box.

        When comparing with a BoundingBox, the channel axis is ignored
        and only the spatial (x, y, z) dimensions are compared.
        """
        from .bounding_box import BoundingBox

        if isinstance(other, BoundingBox):
            denormalized = self.denormalize()
            if denormalized is not self:
                return denormalized == other
            # Cannot denormalize to BoundingBox, axes are incompatible
            return False
        return super().__eq__(other)

    def _denormalize_for_bbox_op(
        self, other: NDBoundingBox
    ) -> tuple[NDBoundingBox, NDBoundingBox, int] | None:
        """Helper for operations with BoundingBox.

        If other is a BoundingBox and self can be denormalized to a BoundingBox,
        returns (denormalized_self, other, num_channels).
        Returns None if the operation should proceed with the default behavior.
        """
        from .bounding_box import BoundingBox

        if isinstance(other, BoundingBox):
            denormalized = self.denormalize()
            if denormalized is not self:
                num_channels = self.size.c
                return (denormalized, other, num_channels)
        return None

    def intersected_with(
        self, other: NDBoundingBox, dont_assert: bool = False
    ) -> "NormalizedBoundingBox":
        """Returns the intersection of two bounding boxes.

        When intersecting with a BoundingBox, the operation is performed
        on the spatial (x, y, z) dimensions only, ignoring the channel axis.
        The result is a NormalizedBoundingBox with the channel count preserved.
        """
        denormalized = self._denormalize_for_bbox_op(other)
        if denormalized is not None:
            result = denormalized[0].intersected_with(denormalized[1], dont_assert)
            return result.normalize_axes(denormalized[2])
        return super().intersected_with(other, dont_assert)  # type: ignore[arg-type]

    def extended_by(self, other: NDBoundingBox) -> "NormalizedBoundingBox":
        """Returns the smallest bounding box that contains both bounding boxes.

        When extending with a BoundingBox, the operation is performed
        on the spatial (x, y, z) dimensions only, ignoring the channel axis.
        The result is a NormalizedBoundingBox with the channel count preserved.
        """
        denormalized = self._denormalize_for_bbox_op(other)
        if denormalized is not None:
            result = denormalized[0].extended_by(denormalized[1])
            return result.normalize_axes(denormalized[2])
        return super().extended_by(other)  # type: ignore[arg-type]

    def contains_bbox(self, inner_bbox: NDBoundingBox) -> bool:
        """Check whether a bounding box is completely inside this bounding box.

        When checking containment of a BoundingBox, only the spatial
        (x, y, z) dimensions are considered, ignoring the channel axis.
        """
        denormalized = self._denormalize_for_bbox_op(inner_bbox)
        if denormalized is not None:
            return denormalized[0].contains_bbox(denormalized[1])
        return super().contains_bbox(inner_bbox)  # type: ignore[arg-type]

    @classmethod
    def from_wkw_dict(cls, bbox: dict) -> NDBoundingBox:
        """
        Create an instance of NDBoundingBox from a dictionary representation.

        Args:
            bbox (Dict): The dictionary representation of the bounding box.

        Returns:
            NDBoundingBox: An instance of NDBoundingBox (or a subclass).

        Raises:
            AssertionError: If additionalAxes are present but axisOrder is not provided.
        """
        num_channels = bbox.get("numChannels", 1)
        if (
            bbox.get("channelIndex", 0) == 0
            and ("additionalAxes" not in bbox or bbox["additionalAxes"] == [])
            and ("axisOrder" not in bbox or bbox["axisOrder"] == _DEFAULT_AXIS_ORDER)
        ):
            # Delegate to BoundingBox.from_wkw_dict, if only 3d
            from .bounding_box import BoundingBox

            return BoundingBox(
                bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]]
            ).normalize_axes(num_channels)

        axes = [
            Axis(
                name="c",
                min=bbox.get("channelIndex", 0),
                size=num_channels,
                index=0,
            ),
            Axis(name="x", min=bbox["topLeft"][0], size=bbox["width"], index=1),
            Axis(name="y", min=bbox["topLeft"][1], size=bbox["height"], index=2),
            Axis(name="z", min=bbox["topLeft"][2], size=bbox["depth"], index=3),
        ]

        if "additionalAxes" in bbox:
            assert "axisOrder" in bbox, (
                "If there are additionalAxes an axisOrder needs to be provided."
            )
            axes.extend(
                Axis(
                    name=axis["name"],
                    min=axis["bounds"][0],
                    size=axis["bounds"][1] - axis["bounds"][0],
                    index=axis["index"],
                )
                for axis in bbox["additionalAxes"]
            )

        if "axisOrder" in bbox:
            for axis_name, axis_index in bbox["axisOrder"].items():
                idx = _find_index_by_name(axes, axis_name)
                axes[idx] = attr.evolve(axes[idx], index=axis_index)

        axes = [
            axis
            for axis in axes
            if axis.name in bbox.get("axisOrder", {}).keys()  # in axisOrder
            or any(
                a["name"] == axis.name for a in bbox.get("additionalAxes", [])
            )  # or in additionalAxes
        ]
        axes = sorted(axes, key=lambda axis: axis.index)

        topleft = [axis.min for axis in axes]
        size = [axis.size for axis in axes]
        axis_names = [axis.name for axis in axes]
        index = [axis.index for axis in axes]

        return cls(
            topleft=VecInt(topleft, axes=axis_names),
            size=VecInt(size, axes=axis_names),
            axes=axis_names,
            index=VecInt(index, axes=axis_names),
        )

    def to_wkw_dict(self) -> dict:
        """
        Converts the bounding box object to a json dictionary.

        Returns:
            dict: A json dictionary representing the bounding box.
        """
        if self.axes == ("c", "x", "y", "z"):
            return {
                "topLeft": self.topleft_xyz.to_list(),
                "width": self.size.x,
                "height": self.size.y,
                "depth": self.size.z,
                "numChannels": self.size.c,
            }

        topleft = [None, None, None]
        width, height, depth = None, None, None
        additional_axes = []
        for i, axis in enumerate(self.axes):
            index = self.index[i]
            if axis == "x":
                topleft[0] = self.topleft[index]
                width = self.size[index]
            elif axis == "y":
                topleft[1] = self.topleft[index]
                height = self.size[index]
            elif axis == "z":
                topleft[2] = self.topleft[index]
                depth = self.size[index]
            elif axis == "c":
                pass
            else:
                additional_axes.append(
                    {
                        "name": axis,
                        "bounds": [self.topleft[index], self.bottomright[index]],
                        "index": index,
                    }
                )
        out = {
            "topLeft": topleft,
            "width": width,
            "height": height,
            "depth": depth,
        }
        if additional_axes:
            out["additionalAxes"] = additional_axes

        def _axis_order_field(axes: tuple[str, ...], index: VecInt) -> dict[str, int]:
            axis_order = {axis: index[i] for i, axis in enumerate(axes)}
            ndim = len(axes)
            # Include only axes that are default axes (c, x, y, z) and not in the
            # default axis order. The default axis order is c, x, y, z from the
            # back (z is the last, c is the 4th last).
            minimal_axis_order = {
                axis: index
                for axis, index in axis_order.items()
                if axis not in _DEFAULT_AXIS_ORDER
                or (ndim - len(_DEFAULT_AXIS_ORDER) - _DEFAULT_AXIS_ORDER[axis])
                != index
            }
            return minimal_axis_order

        axis_order = _axis_order_field(self.axes, self.index)
        if len(axis_order) > 0:
            out["axisOrder"] = axis_order

        if "c" in self.axes and self.topleft.c != 0:
            out["channelIndex"] = self.topleft.c
        if "c" in self.axes:
            out["numChannels"] = self.size.c

        return out

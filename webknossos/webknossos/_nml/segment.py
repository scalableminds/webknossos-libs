from typing import NamedTuple, Optional
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from ..geometry import Vec3Int
from .utils import Vector4, enforce_not_null, filter_none_values


class Segment(NamedTuple):
    id: int
    name: Optional[str]
    anchor_position: Optional[Vec3Int]
    color: Optional[Vector4]

    def _dump(self, xf: XmlWriter) -> None:
        if self.anchor_position is None:
            anchor_position_dict = {}
        else:
            anchor_position_dict = {
                "anchorPositionX": str(self.anchor_position.x),
                "anchorPositionY": str(self.anchor_position.y),
                "anchorPositionZ": str(self.anchor_position.z),
            }
        if self.color is None:
            color_dict = {}
        else:
            color_dict = {
                "color.r": str(self.color[0]),
                "color.g": str(self.color[1]),
                "color.b": str(self.color[2]),
                "color.a": str(self.color[3]),
            }
        xf.tag(
            "segment",
            filter_none_values(
                {
                    "id": str(self.id),
                    "name": self.name,
                    **anchor_position_dict,
                    **color_dict,
                }
            ),
        )

    @classmethod
    def _parse(cls, nml_segment: Element) -> "Segment":
        if nml_segment.get("color.r") is None:
            color = None
        else:
            color = (
                float(enforce_not_null(nml_segment.get("color.r"))),
                float(enforce_not_null(nml_segment.get("color.g"))),
                float(enforce_not_null(nml_segment.get("color.b"))),
                float(enforce_not_null(nml_segment.get("color.a"))),
            )
        if nml_segment.get("anchorPositionX") is None:
            anchor_position = None
        else:
            anchor_position = Vec3Int(
                int(enforce_not_null(nml_segment.get("anchorPositionX"))),
                int(enforce_not_null(nml_segment.get("anchorPositionY"))),
                int(enforce_not_null(nml_segment.get("anchorPositionZ"))),
            )
        return cls(
            int(enforce_not_null(nml_segment.get("id"))),
            name=nml_segment.get("name"),
            anchor_position=anchor_position,
            color=color,
        )

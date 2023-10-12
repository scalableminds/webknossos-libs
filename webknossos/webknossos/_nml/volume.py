from typing import List, NamedTuple, Optional
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .segment import Segment
from .utils import enforce_not_null, filter_none_values


class Volume(NamedTuple):
    id: int
    location: Optional[
        str
    ]  # path to a ZIP file containing a wK volume annotation, may be omitted when using skip_volume_data
    # name of an already existing wK volume annotation segmentation layer:
    fallback_layer: Optional[str]
    # older wk versions did not serialize the name which is why the name is optional:
    name: Optional[str]
    segments: List[Segment]
    format: Optional[str] = None
    largest_segment_id: Optional[int] = None

    def _dump(self, xf: XmlWriter) -> None:
        xf.startTag(
            "volume",
            filter_none_values(
                {
                    "id": str(self.id),
                    "location": self.location,
                    "fallbackLayer": self.fallback_layer,
                    "name": self.name,
                    "format": self.format,
                    "largestSegmentId": str(self.largest_segment_id)
                    if self.largest_segment_id
                    else None,
                }
            ),
        )
        if self.segments is not None:
            xf.startTag("segments")
            for segment in self.segments:
                segment._dump(xf)
            xf.endTag()  # segments
        xf.endTag()  # volume

    @classmethod
    def _parse(cls, nml_volume: Element) -> "Volume":
        largest_segment_id_str = nml_volume.get("largestSegmentId", default=None)
        return cls(
            id=int(enforce_not_null(nml_volume.get("id"))),
            location=nml_volume.get("location"),
            fallback_layer=nml_volume.get("fallbackLayer", default=None),
            name=nml_volume.get("name", default=None),
            format=nml_volume.get("format", default=None),
            segments=[],
            largest_segment_id=None
            if largest_segment_id_str is None
            else int(largest_segment_id_str),
        )

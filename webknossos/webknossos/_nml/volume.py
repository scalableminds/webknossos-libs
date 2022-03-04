from typing import NamedTuple, Optional
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import enforce_not_null, filter_none_values


class Volume(NamedTuple):
    id: int
    location: str  # path to a ZIP file containing a wK volume annotation
    fallback_layer: Optional[
        str
    ] = None  # name of an already existing wK volume annotation segmentation layer
    name: Optional[
        str
    ] = None  # older wk versions did not serialize the name which is why the name is optional

    def _dump(self, xf: XmlWriter) -> None:
        xf.tag(
            "volume",
            filter_none_values(
                {
                    "id": str(self.id),
                    "location": self.location,
                    "fallbackLayer": self.fallback_layer,
                    "name": self.name,
                }
            ),
        )

    @classmethod
    def _parse(cls, nml_volume: Element) -> "Volume":
        return cls(
            int(enforce_not_null(nml_volume.get("id"))),
            enforce_not_null(nml_volume.get("location")),
            nml_volume.get("fallbackLayer", default=None),
            nml_volume.get("name", default=None),
        )

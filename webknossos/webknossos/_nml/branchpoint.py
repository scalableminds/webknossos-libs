from typing import NamedTuple, Optional
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import as_int_unless_none, enforce_not_null


class Branchpoint(NamedTuple):
    id: int
    time: Optional[int]  # unix timestamp

    def _dump(self, xf: XmlWriter) -> None:
        props = {"id": str(self.id)}
        if self.time is not None:
            props["time"] = str(self.time)
        xf.tag(
            "branchpoint",
            props,
        )

    @classmethod
    def _parse(cls, nml_branchpoint: Element) -> "Branchpoint":
        return cls(
            int(enforce_not_null(nml_branchpoint.get("id"))),
            as_int_unless_none(nml_branchpoint.get("time")),
        )

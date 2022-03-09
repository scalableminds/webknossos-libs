from typing import NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import enforce_not_null


class Edge(NamedTuple):
    source: int
    target: int

    def _dump(self, xf: XmlWriter) -> None:
        xf.tag("edge", {"source": str(self.source), "target": str(self.target)})

    @classmethod
    def _parse(cls, nml_edge: Element) -> "Edge":
        return cls(
            source=int(enforce_not_null(nml_edge.get("source"))),
            target=int(enforce_not_null(nml_edge.get("target"))),
        )

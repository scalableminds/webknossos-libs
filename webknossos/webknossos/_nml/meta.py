from typing import NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import enforce_not_null


class Meta(NamedTuple):
    name: str
    content: str

    def _dump(self, xf: XmlWriter) -> None:
        xf.tag(
            "meta",
            {"name": self.name, "content": self.content},
        )

    @classmethod
    def _parse(cls, nml_meta: Element) -> "Meta":
        return cls(
            enforce_not_null(nml_meta.get("name", default=None)),
            enforce_not_null(nml_meta.get("content", default=None)),
        )

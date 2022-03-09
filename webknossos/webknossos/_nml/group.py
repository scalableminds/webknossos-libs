from typing import List, NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import enforce_not_null


class Group(NamedTuple):
    id: int
    name: str
    children: List["Group"]  # type: ignore[misc]

    def _dump(self, xf: XmlWriter) -> None:
        xf.startTag("group", {"id": str(self.id), "name": self.name})
        for subgroup in self.children:
            subgroup._dump(xf)
        xf.endTag()  # group

    @classmethod
    def _parse(cls, nml_group: Element) -> "Group":
        return cls(
            int(enforce_not_null(nml_group.get("id"))),
            enforce_not_null(nml_group.get("name")),
            [],
        )

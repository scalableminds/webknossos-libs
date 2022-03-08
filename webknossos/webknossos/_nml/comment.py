from typing import NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import enforce_not_null


class Comment(NamedTuple):
    node: int
    content: str  # free text field, supports Markdown formatting

    def _dump(self, xf: XmlWriter) -> None:
        xf.tag("comment", {"node": str(self.node), "content": self.content})

    @classmethod
    def _parse(cls, nml_comment: Element) -> "Comment":
        return cls(
            int(enforce_not_null(nml_comment.get("node"))),
            enforce_not_null(nml_comment.get("content")),
        )

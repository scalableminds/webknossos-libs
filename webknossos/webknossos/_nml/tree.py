from typing import NamedTuple
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .edge import Edge
from .metadata_entry import MetadataEntry
from .node import Node
from .utils import Vector4, enforce_not_null


class Tree(NamedTuple):
    #  nml-Thing pendant

    id: int
    color: Vector4 | None  # RGBA
    name: str
    nodes: list[Node]
    edges: list[Edge]
    metadata: list[MetadataEntry]
    groupId: int | None = None

    def _dump(self, xf: XmlWriter) -> None:
        color = self.color or (1, 1, 1, 1)
        attributes = {
            "id": str(self.id),
            "color.r": str(color[0]),
            "color.g": str(color[1]),
            "color.b": str(color[2]),
            "color.a": str(color[3]),
            "name": self.name,
        }

        if self.groupId is not None:
            attributes["groupId"] = str(self.groupId)

        xf.startTag("thing", attributes)
        xf.startTag("nodes")
        for node in self.nodes:
            node._dump(xf)
        xf.endTag()  # nodes
        xf.startTag("edges")
        for edge in self.edges:
            edge._dump(xf)
        xf.endTag()  # edges
        xf.startTag("metadata")
        for metadata_entry in self.metadata:
            metadata_entry._dump(xf)
        xf.endTag()  # metadata
        xf.endTag()  # thing

    @classmethod
    def _parse(cls, nml_tree: Element) -> "Tree":
        name = nml_tree.get("name", "") or nml_tree.get("comment", "")

        color = None
        if nml_tree.get("color.r"):  # also checks for empty strings
            color = (
                float(enforce_not_null(nml_tree.get("color.r"))),
                float(enforce_not_null(nml_tree.get("color.g"))),
                float(enforce_not_null(nml_tree.get("color.b"))),
                float(enforce_not_null(nml_tree.get("color.a"))),
            )
        if nml_tree.get("colorr"):
            color = (
                float(enforce_not_null(nml_tree.get("colorr"))),
                float(enforce_not_null(nml_tree.get("colorg"))),
                float(enforce_not_null(nml_tree.get("colorb"))),
                float(enforce_not_null(nml_tree.get("colora"))),
            )
        try:
            groupId = int(nml_tree.get("groupId", default=-1))
        except ValueError:
            groupId = -1

        return cls(
            nodes=[],
            edges=[],
            metadata=[],
            id=int(enforce_not_null(nml_tree.get("id"))),
            name=name,
            groupId=groupId if groupId >= 0 else None,
            color=color,
        )

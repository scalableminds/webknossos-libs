import xml.etree.ElementTree as ET
from typing import IO, BinaryIO, List, NamedTuple, Optional

from loxun import XmlWriter

from .branchpoint import Branchpoint
from .comment import Comment
from .edge import Edge
from .group import Group
from .meta import Meta
from .node import Node
from .parameters import Parameters
from .tree import Tree
from .volume import Volume


class Nml(NamedTuple):
    meta: List[Meta]
    parameters: Parameters  # metadata for annotations
    trees: List[Tree]
    branchpoints: List[Branchpoint]
    comments: List[Comment]
    groups: List[Group]
    volumes: List[
        Volume
    ] = []  # reference to any volume data that is part of this annotation

    def get_meta(self, key: str) -> Optional[str]:
        for entry in self.meta:
            if entry.name == key:
                return entry.content
        return None

    def write(self, file: BinaryIO) -> None:
        """
        Writes the NML object to a file on disk.

        Arguments:
            file (BinaryIO): A Python file handle

        Example:
            ```
            with open("out.nml", "wb") as f:
                nml.write(f, nml)
            ```
        """
        with XmlWriter(file) as xf:
            self._dump(xf)

    def _dump(self, xf: XmlWriter) -> None:
        xf.startTag("things")

        self.parameters._dump(xf)

        for meta_i in self.meta:
            meta_i._dump(xf)

        for tree in self.trees:
            tree._dump(xf)

        xf.startTag("branchpoints")
        for branchpoint in self.branchpoints:
            branchpoint._dump(xf)
        xf.endTag()  # branchpoints

        xf.startTag("comments")
        for comment in self.comments:
            comment._dump(xf)
        xf.endTag()  # comments

        xf.startTag("groups")
        for group in self.groups:
            group._dump(xf)
        xf.endTag()  # groups

        for volume in self.volumes:
            volume._dump(xf)

        xf.endTag()  # things

    @classmethod
    def parse(cls, file: IO[bytes]) -> "Nml":
        """
        Reads a webKnossos NML skeleton file from disk, _parses it and returns an NML Python object

        Arguments:
            file (BinaryIO): A Python file handle

        Return:
            NML: A webKnossos skeleton annotation as Python NML object

        Example:
            ```
            with open("input.nml", "rb") as f:
                nml = Nml.parse(f)
            ```
        """

        meta = []
        parameters = None
        trees = []
        branchpoints = []
        comments = []
        current_tree = None
        root_group = Group(-1, "", [])
        group_stack = [root_group]
        element_stack = []
        volumes = []

        for event, elem in ET.iterparse(file, events=("start", "end")):
            if event == "start":
                element_stack.append(elem)
                if elem.tag == "thing":
                    current_tree = Tree._parse(elem)
                    trees.append(current_tree)
                elif elem.tag == "node":
                    assert (
                        current_tree is not None
                    ), "<node ...> tag needs to be child of a <thing ...> tag."
                    current_tree.nodes.append(Node._parse(elem))
                elif elem.tag == "edge":
                    assert (
                        current_tree is not None
                    ), "<edge ...> tag needs to be child of a <thing ...> tag."
                    current_tree.edges.append(Edge._parse(elem))
                elif elem.tag == "branchpoint":
                    branchpoints.append(Branchpoint._parse(elem))
                elif elem.tag == "comment":
                    comments.append(Comment._parse(elem))
                elif elem.tag == "volume":
                    volumes.append(Volume._parse(elem))
                elif elem.tag == "meta":
                    meta.append(Meta._parse(elem))
                elif elem.tag == "group":
                    group = Group._parse(elem)
                    group_stack[-1].children.append(group)
                    group_stack.append(group)
            elif event == "end":
                if elem.tag == "parameters":
                    parameters = Parameters._parse(elem)
                elif elem.tag == "thing":
                    current_tree = None
                elif elem.tag == "group":
                    group_stack.pop()

                element_stack.pop()
                # Do not clear the elements of the parameters tag as we want to _parse those all at once
                # when the closing parameters tag is _parsed
                if len(element_stack) and element_stack[-1].tag != "parameters":
                    # Discard the element to save memory
                    elem.clear()

        assert parameters is not None, "No parameters found in NML"

        return cls(
            meta=meta,
            parameters=parameters,
            trees=trees,
            branchpoints=branchpoints,
            comments=comments,
            groups=root_group.children,
            volumes=volumes,
        )

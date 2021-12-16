import xml.etree.ElementTree as ET
from typing import BinaryIO, List, NamedTuple, Optional, Text, Tuple, TypeVar
from xml.etree.ElementTree import Element

from loxun import XmlWriter

Vector3 = Tuple[float, float, float]
Vector4 = Tuple[float, float, float, float]
IntVector6 = Tuple[int, int, int, int, int, int]

T = TypeVar("T")


def enforce_not_null(val: Optional[T]) -> T:
    if val is None:
        raise ValueError("Value is None")
    return val


def as_int_unless_none(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    return int(val)


class NMLParameters(NamedTuple):
    """
    Contains common metadata for NML files

    Note:
        Setting a task bounding boxes will cause wK to 1) render these visually and 2) prevent data loading from outside them.

    Attributes:
        name (str): Name of a dataset that the annotation is based on. Will cause wK to open the given skeleton annotation with the referenced dataset.
        scale (Vector3[float]): Voxel scale of the referenced dataset in nanometers.
        offset (Optional[Vector3[float]]): Deprecated. Kept for backward compatibility.
        time (Optional[int]): A UNIX timestamp marking the creation time & date of an annotation.
        editPosition (Optional[Vector3[float]]): The position of the wK camera when creating/downloading an annotation
        editRotation (Optional[Vector3[float]]): The rotation of the wK camera when creating/downloading an annotation
        zoomLevel (Optional[float]): The zoomLevel of the wK camera when creating/downloading an annotation
        taskBoundingBox (Optional[IntVector6[int]]): A custom bounding box specified as part of a [wK task](https://docs.webknossos.org/webknossos/tasks.html). Will be rendered in wK.
        userBoundingBoxes (Optional[List[IntVector6[int]]]): A list of custom user-defined bounding boxes. Will be rendered in wK.
    """

    name: str
    scale: Vector3
    offset: Optional[Vector3] = None
    time: Optional[int] = None
    editPosition: Optional[Vector3] = None
    editRotation: Optional[Vector3] = None
    zoomLevel: Optional[float] = None
    taskBoundingBox: Optional[IntVector6] = None
    userBoundingBoxes: Optional[List[IntVector6]] = None


class Node(NamedTuple):
    """
    A webKnossos skeleton node annotation object.

    Attributes:
        id (int): A unique identifier
        position (Vector3): 3D position of a node. Format: [x, y, z]
        radius (Optional[float]): Radius of a node when rendered in wK. Unit: nanometers (nm)
        rotation (Optional[Vector3]): 3D rotation of the camera when the node was annotated. Mostly relevant for `Flight` mode to resume in the same direction when returning to `Flight` mode.
        inVp (Optional[int]): Enumeration of the wK UI viewport in which the node was annotated. `0`: XY plane, `1`: YZ plane. `2`: XY plane, `3`: 3D viewport
        inMag (Optional[int]): wK rendering magnification-level when the node was annotated. Lower magnification levels typically indicate a "zoomed-in" workflow resulting in more accurate annotations.
        bitDepth (Optional[int]): wK rendering bit-depth when the node was annotated. 4bit (lower data quality) or 8bit (regular quality). Lower quality data rendering might lead to less accurate annotations.
        interpolation (Optional[bool]): wK rendering interpolation flag when the node was annotated. Interpolated data rendering might lead to less accurate annotations.
        time (Optional[int]): A Unix timestamp marking the creation time of the node.

    """

    id: int
    position: Vector3
    radius: Optional[float] = None
    rotation: Optional[Vector3] = None
    inVp: Optional[int] = None
    inMag: Optional[int] = None
    bitDepth: Optional[int] = None
    interpolation: Optional[bool] = None
    time: Optional[int] = None


class Edge(NamedTuple):
    """
    A webKnossos skeleton edge.

    Attributes:
        source (int): node id reference
        target (int): node id reference
    """

    source: int
    target: int


class Tree(NamedTuple):
    """
    A webKnossos skeleton (tree) object. A graph structure consisting of nodes and edges.

    Attributes:
        id: int
        color (Vector4): RGBA
        name: str
        nodes: List[Node]
        edges: List[Edge]
        groupId (Optional[int]): group id reference
    """

    id: int
    color: Optional[Vector4]
    name: str
    nodes: List[Node]
    edges: List[Edge]
    groupId: Optional[int] = None


class Branchpoint(NamedTuple):
    """
    A webKnossos branchpoint, i.e. a skeleton node with more than one outgoing edge.

    Attributes:
        id (int): Reference to a `Node` ID
        time (int): Unix timestamp
    """

    id: int
    time: Optional[int]


class Group(NamedTuple):
    """
    A container to group several skeletons (trees) together. Mostly for cosmetic or organizational purposes.

    Attributes:
        id (int): A unique group identifier
        name (str): Name of the group. Will be displayed in wK UI
        children (List[Group]): List of all sub-groups belonging to this parent element for nested structures
    """

    id: int
    name: str
    children: List["Group"]  # type: ignore


class Comment(NamedTuple):
    """
    A single comment belonging to a skeleton node.

    Attributes:
        node (int): Reference to a `Node` ID
        content (str): A free text field. Supports Markdown formatting.
    """

    node: int
    content: str


class Volume(NamedTuple):
    """
    A metadata reference to a wK volume annotation. Typically, the volume annotation data is provided in a ZIP file in the same directory as the skeleton annotation.

    Attributes:
        id (int): A unique identifier
        location (str): A path to a ZIP file containing a wK volume annotation
        fallback_layer (Optional[str]): Name of an already existing wK volume annotation segmentation layer (aka "fallback layer")
    """

    id: int
    location: str
    fallback_layer: Optional[str] = None


class NML(NamedTuple):
    """
    A complete webKnossos skeleton annotation object contain one or more skeletons (trees).

    Attributes:
        parameters (NMLParameters): All the metadata attributes associated with a wK annotation.
        trees (List[Tree]): A list of all skeleton/tree objects. Usually contains the majority of the annotated skeleton information.
        branchpoints (List[Branchpoint]): A list of all branchpoint objects.
        comments (List[Comment]): A list of all comment objects.
        groups (List[Group]): A list of all group objects.
        volume (Optional[Volume]): A reference to any volume data that is part of this annotation.
    """

    parameters: NMLParameters
    trees: List[Tree]
    branchpoints: List[Branchpoint]
    comments: List[Comment]
    groups: List[Group]
    volume: Optional[Volume] = None


def __parse_user_bounding_boxes(nml_parameters: Element) -> Optional[List[IntVector6]]:
    # ToDo support color, id, name, isVisible attributes pylint: disable=fixme
    # https://github.com/scalableminds/wknml/issues/46
    if nml_parameters.find("userBoundingBox") is None:
        return None
    bb_elements = nml_parameters.findall("userBoundingBox")
    return [__parse_bounding_box(bb_element) for bb_element in bb_elements]


def __parse_task_bounding_box(nml_parameters: Element) -> Optional[IntVector6]:
    bb_element = nml_parameters.find("taskBoundingBox")
    if bb_element is not None:
        return __parse_bounding_box(bb_element)

    return None


def __parse_bounding_box(bounding_box_element: Element) -> IntVector6:
    return (
        int(bounding_box_element.get("topLeftX", 0)),
        int(bounding_box_element.get("topLeftY", 0)),
        int(bounding_box_element.get("topLeftZ", 0)),
        int(bounding_box_element.get("width", 0)),
        int(bounding_box_element.get("height", 0)),
        int(bounding_box_element.get("depth", 0)),
    )


def __parse_parameters(nml_parameters: Element) -> NMLParameters:
    offset = None
    if nml_parameters.find("offset") is not None:
        offset_element = enforce_not_null(nml_parameters.find("offset"))
        offset = (
            float(offset_element.get("x", 0)),
            float(offset_element.get("y", 0)),
            float(offset_element.get("z", 0)),
        )

    editRotation = None
    if nml_parameters.find("editRotation") is not None:
        edit_rotation_element = enforce_not_null(nml_parameters.find("editRotation"))
        editRotation = (
            float(edit_rotation_element.get("xRot", 0)),
            float(edit_rotation_element.get("yRot", 0)),
            float(edit_rotation_element.get("zRot", 0)),
        )

    editPosition = None
    if nml_parameters.find("editPosition") is not None:
        editPosition = (
            float(enforce_not_null(nml_parameters.find("editPosition")).get("x", 0)),
            float(enforce_not_null(nml_parameters.find("editPosition")).get("y", 0)),
            float(enforce_not_null(nml_parameters.find("editPosition")).get("z", 0)),
        )

    time = None
    if nml_parameters.find("time") is not None:
        time = int(enforce_not_null(nml_parameters.find("time")).get("ms", 0))

    zoomLevel: Optional[float] = None
    if nml_parameters.find("zoomLevel") is not None:
        zoom_str = enforce_not_null(nml_parameters.find("zoomLevel")).get("zoom", 0)
        try:
            zoomLevel = int(zoom_str)
        except ValueError:
            zoomLevel = float(zoom_str)

    taskBoundingBox = __parse_task_bounding_box(nml_parameters)
    userBoundingBoxes = __parse_user_bounding_boxes(nml_parameters)

    scale_element = enforce_not_null(nml_parameters.find("scale"))
    return NMLParameters(
        name=enforce_not_null(nml_parameters.find("experiment")).get(
            "name", "Unnamed Experiment"
        ),
        scale=(
            float(scale_element.get("x", 0)),
            float(scale_element.get("y", 0)),
            float(scale_element.get("z", 0)),
        ),
        offset=offset,
        time=time,
        editPosition=editPosition,
        editRotation=editRotation,
        zoomLevel=zoomLevel,
        taskBoundingBox=taskBoundingBox,
        userBoundingBoxes=userBoundingBoxes,
    )


def __parse_node(nml_node: Element) -> Node:
    rotation = None
    if nml_node.get("rotX") is not None:
        rotation = (
            float(enforce_not_null(nml_node.get("rotX"))),
            float(enforce_not_null(nml_node.get("rotY"))),
            float(enforce_not_null(nml_node.get("rotZ"))),
        )

    return Node(
        id=int(enforce_not_null(nml_node.get("id"))),
        radius=float(nml_node.get("radius", 0))
        if nml_node.get("radius") is not None
        else None,
        position=(
            float(nml_node.get("x", 0)),
            float(nml_node.get("y", 0)),
            float(nml_node.get("z", 0)),
        ),
        rotation=rotation,
        inVp=int(nml_node.get("inVp", 0)) if nml_node.get("inVp") is not None else None,
        inMag=int(nml_node.get("inMag", 0))
        if nml_node.get("inMag") is not None
        else None,
        bitDepth=int(nml_node.get("bitDepth", 0))
        if nml_node.get("bitDepth") is not None
        else None,
        interpolation=bool(nml_node.get("interpolation"))
        if nml_node.get("interpolation") is not None
        else None,
        time=int(nml_node.get("time", 0)) if nml_node.get("time") is not None else None,
    )


def __parse_edge(nml_edge: Element) -> Edge:
    return Edge(
        source=int(enforce_not_null(nml_edge.get("source"))),
        target=int(enforce_not_null(nml_edge.get("target"))),
    )


def __parse_tree(nml_tree: Element) -> Tree:
    name = nml_tree.get("name", "") or nml_tree.get("comment", "")

    color = None
    if "color.r" in nml_tree.attrib:
        color = (
            float(enforce_not_null(nml_tree.get("color.r"))),
            float(enforce_not_null(nml_tree.get("color.g"))),
            float(enforce_not_null(nml_tree.get("color.b"))),
            float(enforce_not_null(nml_tree.get("color.a"))),
        )
    if "colorr" in nml_tree.attrib:
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

    return Tree(
        nodes=[],
        edges=[],
        id=int(enforce_not_null(nml_tree.get("id"))),
        name=name,
        groupId=groupId if groupId >= 0 else None,
        color=color,
    )


def __parse_branchpoint(nml_branchpoint: Element) -> Branchpoint:
    return Branchpoint(
        int(enforce_not_null(nml_branchpoint.get("id"))),
        as_int_unless_none(nml_branchpoint.get("time")),
    )


def __parse_comment(nml_comment: Element) -> Comment:
    return Comment(
        int(enforce_not_null(nml_comment.get("node"))),
        enforce_not_null(nml_comment.get("content")),
    )


def __parse_group(nml_group: Element) -> Group:
    return Group(
        int(enforce_not_null(nml_group.get("id"))),
        enforce_not_null(nml_group.get("name")),
        [],
    )


def __parse_volume(nml_volume: Element) -> Volume:
    return Volume(
        int(enforce_not_null(nml_volume.get("id"))),
        enforce_not_null(nml_volume.get("location")),
        nml_volume.get("fallback_layer", default=None),
    )


def parse_nml(file: BinaryIO) -> NML:
    """
    Reads a webKnossos NML skeleton file from disk, parses it and returns an NML Python object

    Arguments:
        file (BinaryIO): A Python file handle

    Return:
        NML: A webKnossos skeleton annotation as Python NML object

    Example:
        ```
        with open("input.nml", "rb") as f:
            nml = wknml.parse_nml(f, nml)
        ```
    """

    parameters = None
    trees = []
    branchpoints = []
    comments = []
    current_tree = None
    root_group = Group(-1, "", [])
    group_stack = [root_group]
    element_stack = []
    volume = None

    for event, elem in ET.iterparse(file, events=("start", "end")):
        if event == "start":
            element_stack.append(elem)
            if elem.tag == "thing":
                current_tree = __parse_tree(elem)
                trees.append(current_tree)
            elif elem.tag == "node":
                assert (
                    current_tree is not None
                ), "<node ...> tag needs to be child of a <thing ...> tag."
                current_tree.nodes.append(__parse_node(elem))
            elif elem.tag == "edge":
                assert (
                    current_tree is not None
                ), "<edge ...> tag needs to be child of a <thing ...> tag."
                current_tree.edges.append(__parse_edge(elem))
            elif elem.tag == "branchpoint":
                branchpoints.append(__parse_branchpoint(elem))
            elif elem.tag == "comment":
                comments.append(__parse_comment(elem))
            elif elem.tag == "volume":
                volume = __parse_volume(elem)
            elif elem.tag == "group":
                group = __parse_group(elem)
                group_stack[-1].children.append(group)
                group_stack.append(group)
        elif event == "end":
            if elem.tag == "parameters":
                parameters = __parse_parameters(elem)
            elif elem.tag == "thing":
                current_tree = None
            elif elem.tag == "group":
                group_stack.pop()

            element_stack.pop()
            # Do not clear the elements of the parameters tag as we want to parse those all at once
            # when the closing parameters tag is parsed
            if len(element_stack) and element_stack[-1].tag != "parameters":
                # Discard the element to save memory
                elem.clear()

    assert parameters is not None, "No parameters found in NML"

    return NML(
        parameters=parameters,
        trees=trees,
        branchpoints=branchpoints,
        comments=comments,
        groups=root_group.children,
        volume=volume,
    )


def __dump_task_bounding_box(xf: XmlWriter, parameters: NMLParameters) -> None:
    task_bounding_box = getattr(parameters, "taskBoundingBox")
    if task_bounding_box is not None:
        __dump_bounding_box(xf, task_bounding_box, "taskBoundingBox")


def __dump_user_bounding_boxes(xf: XmlWriter, parameters: NMLParameters) -> None:
    user_bounding_boxes = getattr(parameters, "userBoundingBoxes")

    if user_bounding_boxes is not None:
        for user_bounding_box in user_bounding_boxes:
            __dump_bounding_box(xf, user_bounding_box, "userBoundingBox")


def __dump_bounding_box(
    xf: XmlWriter, bounding_box: IntVector6, tag_name: Text
) -> None:
    xf.tag(
        tag_name,
        {
            "topLeftX": str(bounding_box[0]),
            "topLeftY": str(bounding_box[1]),
            "topLeftZ": str(bounding_box[2]),
            "width": str(bounding_box[3]),
            "height": str(bounding_box[4]),
            "depth": str(bounding_box[5]),
        },
    )


def __dump_parameters(xf: XmlWriter, parameters: NMLParameters) -> None:
    xf.startTag("parameters")
    xf.tag("experiment", {"name": parameters.name})
    xf.tag(
        "scale",
        {
            "x": str(float(parameters.scale[0])),
            "y": str(float(parameters.scale[1])),
            "z": str(float(parameters.scale[2])),
        },
    )

    if parameters.offset is not None:
        xf.tag(
            "offset",
            {
                "x": str(float(parameters.offset[0])),
                "y": str(float(parameters.offset[1])),
                "z": str(float(parameters.offset[2])),
            },
        )

    if parameters.time is not None:
        xf.tag("time", {"ms": str(parameters.time)})
    if parameters.editPosition is not None:
        xf.tag(
            "editPosition",
            {
                "x": str(float(parameters.editPosition[0])),
                "y": str(float(parameters.editPosition[1])),
                "z": str(float(parameters.editPosition[2])),
            },
        )
    if parameters.editRotation is not None:
        xf.tag(
            "editRotation",
            {
                "xRot": str(float(parameters.editRotation[0])),
                "yRot": str(float(parameters.editRotation[1])),
                "zRot": str(float(parameters.editRotation[2])),
            },
        )
    if parameters.zoomLevel is not None:
        xf.tag("zoomLevel", {"zoom": str(parameters.zoomLevel)})

    __dump_task_bounding_box(xf, parameters)
    __dump_user_bounding_boxes(xf, parameters)

    xf.endTag()  # parameters


def __dump_node(xf: XmlWriter, node: Node) -> None:

    attributes = {
        "id": str(node.id),
        "x": str(float(node.position[0])),
        "y": str(float(node.position[1])),
        "z": str(float(node.position[2])),
    }

    if node.radius is not None:
        attributes["radius"] = str(node.radius)

    if node.rotation is not None:
        attributes["rotX"] = str(node.rotation[0])
        attributes["rotY"] = str(node.rotation[1])
        attributes["rotZ"] = str(node.rotation[2])

    if node.inVp is not None:
        attributes["inVp"] = str(node.inVp)

    if node.inMag is not None:
        attributes["inMag"] = str(node.inMag)

    if node.bitDepth is not None:
        attributes["bitDepth"] = str(node.bitDepth)

    if node.interpolation is not None:
        attributes["interpolation"] = str(node.interpolation)

    if node.time is not None:
        attributes["time"] = str(node.time)

    xf.tag("node", attributes)


def __dump_edge(xf: XmlWriter, edge: Edge) -> None:
    xf.tag("edge", {"source": str(edge.source), "target": str(edge.target)})


def __dump_tree(xf: XmlWriter, tree: Tree) -> None:
    color = tree.color or (1, 1, 1, 1)
    attributes = {
        "id": str(tree.id),
        "color.r": str(color[0]),
        "color.g": str(color[1]),
        "color.b": str(color[2]),
        "color.a": str(color[3]),
        "name": tree.name,
    }

    if tree.groupId is not None:
        attributes["groupId"] = str(tree.groupId)

    xf.startTag("thing", attributes)
    xf.startTag("nodes")
    for n in tree.nodes:
        __dump_node(xf, n)
    xf.endTag()  # nodes
    xf.startTag("edges")
    for e in tree.edges:
        __dump_edge(xf, e)
    xf.endTag()  # edges
    xf.endTag()  # thing


def __dump_branchpoint(xf: XmlWriter, branchpoint: Branchpoint) -> None:
    if branchpoint.time is not None:
        xf.tag(
            "branchpoint", {"id": str(branchpoint.id), "time": str(branchpoint.time)}
        )
    else:
        xf.tag("branchpoint", {"id": str(branchpoint.id)})


def __dump_comment(xf: XmlWriter, comment: Comment) -> None:
    if comment.content is not None:
        xf.tag("comment", {"node": str(comment.node), "content": comment.content})
    else:
        xf.tag("comment", {"node": str(comment.node)})


def __dump_volume(xf: XmlWriter, volume: Optional[Volume]) -> None:
    if volume is not None:
        if volume.fallback_layer is not None:
            xf.tag(
                "volume",
                {
                    "id": str(volume.id),
                    "location": volume.location,
                    "fallbackLayer": volume.fallback_layer,
                },
            )
        else:
            xf.tag(
                "volume",
                {
                    "id": str(volume.id),
                    "location": volume.location,
                },
            )


def __dump_group(xf: XmlWriter, group: Group) -> None:
    xf.startTag("group", {"id": str(group.id), "name": group.name})
    for g in group.children:
        __dump_group(xf, g)
    xf.endTag()  # group


def __dump_nml(xf: XmlWriter, nml: NML) -> None:
    xf.startTag("things")
    __dump_parameters(xf, nml.parameters)
    for t in nml.trees:
        __dump_tree(xf, t)

    xf.startTag("branchpoints")
    for b in nml.branchpoints:
        __dump_branchpoint(xf, b)
    xf.endTag()  # branchpoints

    xf.startTag("comments")
    for c in nml.comments:
        __dump_comment(xf, c)
    xf.endTag()  # comments

    xf.startTag("groups")
    for g in nml.groups:
        __dump_group(xf, g)
    xf.endTag()  # groups

    __dump_volume(xf, nml.volume)

    xf.endTag()  # things


def write_nml(file: BinaryIO, nml: NML) -> None:
    """
    Writes an NML object to a file on disk.

    Arguments:
        file (BinaryIO): A Python file handle
        nml (NML): A NML object that should be persisted to disk

    Example:
        ```
        with open("out.nml", "wb") as f:
            wknml.write_nml(f, nml)
        ```
    """
    with XmlWriter(file) as xf:
        __dump_nml(xf, nml)

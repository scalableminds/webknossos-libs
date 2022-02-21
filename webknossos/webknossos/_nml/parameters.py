from typing import List, NamedTuple, Optional, Text
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import IntVector6, Vector3, enforce_not_null, filter_none_values


class Parameters(NamedTuple):
    name: str  # dataset name
    scale: Vector3  # dataset scale
    description: Optional[str] = None
    organization: Optional[str] = None
    offset: Optional[Vector3] = None  # deprecated. Kept for backward compatibility.
    time: Optional[
        int
    ] = None  # UNIX timestamp marking the creation time & date of an annotation.
    editPosition: Optional[Vector3] = None
    editRotation: Optional[Vector3] = None
    zoomLevel: Optional[float] = None
    taskBoundingBox: Optional[IntVector6] = None
    userBoundingBoxes: Optional[List[IntVector6]] = None

    def _dump_bounding_box(
        self, xf: XmlWriter, bounding_box: IntVector6, tag_name: Text
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

    def _dump_task_bounding_box(self, xf: XmlWriter) -> None:
        task_bounding_box = getattr(self, "taskBoundingBox")
        if task_bounding_box is not None:
            self._dump_bounding_box(xf, task_bounding_box, "taskBoundingBox")

    def _dump_user_bounding_boxes(self, xf: XmlWriter) -> None:
        user_bounding_boxes = getattr(self, "userBoundingBoxes")

        if user_bounding_boxes is not None:
            # pylint: disable=not-an-iterable
            for user_bounding_box in user_bounding_boxes:
                self._dump_bounding_box(xf, user_bounding_box, "userBoundingBox")

    def _dump(self, xf: XmlWriter) -> None:
        xf.startTag("parameters")
        xf.tag(
            "experiment",
            filter_none_values(
                {
                    "name": self.name,
                    "organization": self.organization,
                    "description": self.description,
                }
            ),
        )
        xf.tag(
            "scale",
            {
                "x": str(float(self.scale[0])),
                "y": str(float(self.scale[1])),
                "z": str(float(self.scale[2])),
            },
        )

        if self.offset is not None:
            # pylint: disable=unsubscriptable-object
            xf.tag(
                "offset",
                {
                    "x": str(float(self.offset[0])),
                    "y": str(float(self.offset[1])),
                    "z": str(float(self.offset[2])),
                },
            )

        if self.time is not None:
            xf.tag("time", {"ms": str(self.time)})
        if self.editPosition is not None:
            # pylint: disable=unsubscriptable-object
            xf.tag(
                "editPosition",
                {
                    "x": str(float(self.editPosition[0])),
                    "y": str(float(self.editPosition[1])),
                    "z": str(float(self.editPosition[2])),
                },
            )
        if self.editRotation is not None:
            # pylint: disable=unsubscriptable-object
            xf.tag(
                "editRotation",
                {
                    "xRot": str(float(self.editRotation[0])),
                    "yRot": str(float(self.editRotation[1])),
                    "zRot": str(float(self.editRotation[2])),
                },
            )
        if self.zoomLevel is not None:
            xf.tag("zoomLevel", {"zoom": str(self.zoomLevel)})

        self._dump_task_bounding_box(xf)
        self._dump_user_bounding_boxes(xf)

        xf.endTag()  # parameters

    @classmethod
    def _parse_bounding_box(cls, bounding_box_element: Element) -> IntVector6:
        return (
            int(bounding_box_element.get("topLeftX", 0)),
            int(bounding_box_element.get("topLeftY", 0)),
            int(bounding_box_element.get("topLeftZ", 0)),
            int(bounding_box_element.get("width", 0)),
            int(bounding_box_element.get("height", 0)),
            int(bounding_box_element.get("depth", 0)),
        )

    @classmethod
    def _parse_user_bounding_boxes(
        cls, nml_parameters: Element
    ) -> Optional[List[IntVector6]]:
        # ToDo support color, id, name, isVisible attributes pylint: disable=fixme
        # https://github.com/scalableminds/wknml/issues/46
        if nml_parameters.find("userBoundingBox") is None:
            return None
        bb_elements = nml_parameters.findall("userBoundingBox")
        return [cls._parse_bounding_box(bb_element) for bb_element in bb_elements]

    @classmethod
    def _parse_task_bounding_box(cls, nml_parameters: Element) -> Optional[IntVector6]:
        bb_element = nml_parameters.find("taskBoundingBox")
        if bb_element is not None:
            return cls._parse_bounding_box(bb_element)

        return None

    @classmethod
    def _parse(cls, nml_parameters: Element) -> "Parameters":
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
            edit_rotation_element = enforce_not_null(
                nml_parameters.find("editRotation")
            )
            editRotation = (
                float(edit_rotation_element.get("xRot", 0)),
                float(edit_rotation_element.get("yRot", 0)),
                float(edit_rotation_element.get("zRot", 0)),
            )

        editPosition = None
        if nml_parameters.find("editPosition") is not None:
            editPosition = (
                float(
                    enforce_not_null(nml_parameters.find("editPosition")).get("x", 0)
                ),
                float(
                    enforce_not_null(nml_parameters.find("editPosition")).get("y", 0)
                ),
                float(
                    enforce_not_null(nml_parameters.find("editPosition")).get("z", 0)
                ),
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

        taskBoundingBox = cls._parse_task_bounding_box(nml_parameters)
        userBoundingBoxes = cls._parse_user_bounding_boxes(nml_parameters)

        scale_element = enforce_not_null(nml_parameters.find("scale"))
        experiment_element = enforce_not_null(nml_parameters.find("experiment"))
        return cls(
            name=experiment_element.get("name", "Unnamed Experiment"),
            description=experiment_element.get("description"),
            organization=experiment_element.get("organization"),
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

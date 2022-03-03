from typing import NamedTuple, Optional
from xml.etree.ElementTree import Element

from loxun import XmlWriter

from .utils import Vector3, enforce_not_null


class Node(NamedTuple):
    id: int
    position: Vector3
    radius: Optional[
        float
    ] = None  # radius of a node when rendered in wK, unit: nanometers (nm)
    # 3D rotation of the camera when the node was annotated.
    # Mostly relevant for `Flight` mode to resume in the same direction when returning to `Flight` mode.
    rotation: Optional[Vector3] = None
    inVp: Optional[
        int
    ] = None  # wK UI viewport in which the node was set, `0`: XY plane, `1`: YZ plane. `2`: XY plane, `3`: 3D viewport
    # wK rendering magnification-level when the node was annotated.
    # Lower magnification levels typically indicate a "zoomed-in" workflow resulting in more accurate annotations.
    inMag: Optional[int] = None
    # wK rendering bit-depth when the node was annotated. 4bit (lower data quality) or 8bit (regular quality).
    # Lower quality data rendering might lead to less accurate annotations.
    bitDepth: Optional[int] = None
    # wK rendering interpolation flag when the node was annotated.
    # Interpolated data rendering might lead to less accurate annotations.
    interpolation: Optional[bool] = None
    time: Optional[int] = None  # unix timestamp marking the creation time

    def _dump(self, xf: XmlWriter) -> None:
        attributes = {
            "id": str(self.id),
            "x": str(float(self.position[0])),
            "y": str(float(self.position[1])),
            "z": str(float(self.position[2])),
        }

        if self.radius is not None:
            attributes["radius"] = str(self.radius)

        if self.rotation is not None:
            # pylint: disable=unsubscriptable-object
            attributes["rotX"] = str(self.rotation[0])
            attributes["rotY"] = str(self.rotation[1])
            attributes["rotZ"] = str(self.rotation[2])

        if self.inVp is not None:
            attributes["inVp"] = str(self.inVp)

        if self.inMag is not None:
            attributes["inMag"] = str(self.inMag)

        if self.bitDepth is not None:
            attributes["bitDepth"] = str(self.bitDepth)

        if self.interpolation is not None:
            attributes["interpolation"] = str(self.interpolation)

        if self.time is not None:
            attributes["time"] = str(self.time)

        xf.tag("node", attributes)

    @classmethod
    def _parse(cls, nml_node: Element) -> "Node":
        rotation = None
        if nml_node.get("rotX") is not None:
            rotation = (
                float(enforce_not_null(nml_node.get("rotX"))),
                float(enforce_not_null(nml_node.get("rotY"))),
                float(enforce_not_null(nml_node.get("rotZ"))),
            )

        return cls(
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
            inVp=int(nml_node.get("inVp", 0))
            if nml_node.get("inVp") is not None
            else None,
            inMag=int(nml_node.get("inMag", 0))
            if nml_node.get("inMag") is not None
            else None,
            bitDepth=int(nml_node.get("bitDepth", 0))
            if nml_node.get("bitDepth") is not None
            else None,
            interpolation=bool(nml_node.get("interpolation"))
            if nml_node.get("interpolation") is not None
            else None,
            time=int(nml_node.get("time", 0))
            if nml_node.get("time") is not None
            else None,
        )

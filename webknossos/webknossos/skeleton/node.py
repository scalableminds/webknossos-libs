from typing import TYPE_CHECKING, Any

import attr

from ..geometry import Vec3Int

if TYPE_CHECKING:
    from .skeleton import Skeleton

Vec3Float = tuple[float, float, float]


# Defining an order on nodes is necessary to allow to sort them,
# which is used in the tree's equality check, see Tree.__eq__().
@attr.define(order=True)
class Node:
    """A node in a skeleton tree representing a point in 3D space with additional metadata.

    The Node class represents individual points in a skeleton annotation. Each node has a 3D position
    and can contain additional metadata such as comments, radius, rotation, and viewport information.
    Nodes are typically created and managed through Tree instances rather than directly.

    Attributes:
        position: 3D coordinates of the node as (x, y, z).
        comment: Optional text annotation for the node.
        radius: Optional radius value, useful for representing varying thicknesses.
        rotation: Optional 3D rotation as (x, y, z) in radians.
        inVp: Optional viewport number where the node was created.
        inMag: Optional magnification level at which the node was created.
        bitDepth: Optional bit depth of the data at node creation.
        interpolation: Optional flag indicating if the node was created through interpolation.
        time: Optional timestamp for node creation.
        is_branchpoint: Boolean indicating if this node is a branching point.
        branchpoint_time: Optional timestamp when the node was marked as a branchpoint.

    Notes:
        Nodes should typically be created using `Tree.add_node()` rather than instantiated directly.
        This ensures proper integration with the skeleton structure.

    Examples:
        ```python
        # Create a skeleton and tree
        skeleton = Skeleton(name="example")
        tree = skeleton.add_tree("dendrite")

        # Add nodes and connect them
        node1 = tree.add_node(position=(0, 0, 0), comment="soma")
        node2 = tree.add_node(position=(100, 0, 0), radius=1.5)
        tree.add_edge(node1, node2)

        # Mark as branchpoint if needed
        node1.is_branchpoint = True
        ```
    """

    position: Vec3Int
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False, order=False)
    _id: int = attr.ib(init=False)
    comment: str | None = None
    radius: float | None = None
    rotation: Vec3Float | None = None
    inVp: int | None = None
    inMag: int | None = None
    bitDepth: int | None = None
    interpolation: bool | None = None
    time: int | None = None

    is_branchpoint: bool = False
    branchpoint_time: int | None = None
    _enforced_id: int | None = attr.ib(None, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._skeleton._element_id_generator.__next__()

    @property
    def id(self) -> int:
        return self._id

    def __hash__(self) -> int:
        return self._id

    def get_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "comment": self.comment,
            "radius": self.radius,
            "rotation": self.rotation,
            "inVp": self.inVp,
            "inMag": self.inMag,
            "bitDepth": self.bitDepth,
            "interpolation": self.interpolation,
            "time": self.time,
            "is_branchpoint": self.is_branchpoint,
            "branchpoint_time": self.branchpoint_time,
        }

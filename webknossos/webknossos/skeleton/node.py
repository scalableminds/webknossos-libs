from typing import TYPE_CHECKING, Optional, Tuple

import attr

if TYPE_CHECKING:
    from webknossos.skeleton import Skeleton

Vector3 = Tuple[float, float, float]


@attr.define()
class Node:
    position: Vector3
    _skeleton: "Skeleton" = attr.ib(eq=False, repr=False)
    _id: int = attr.ib(init=False)
    comment: Optional[str] = None
    radius: Optional[float] = None
    rotation: Optional[Vector3] = None
    inVp: Optional[int] = None
    inMag: Optional[int] = None
    bitDepth: Optional[int] = None
    interpolation: Optional[bool] = None
    time: Optional[int] = None

    is_branchpoint: bool = False
    branchpoint_time: Optional[int] = None
    _enforced_id: Optional[int] = attr.ib(None, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._enforced_id is not None:
            self._id = self._enforced_id
        else:
            self._id = self._skeleton.element_id_generator.__next__()

    @property
    def id(self) -> int:
        return self._id

    def __hash__(self) -> int:
        return self._id

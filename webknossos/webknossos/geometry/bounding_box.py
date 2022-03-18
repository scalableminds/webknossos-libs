import json
import re
from collections import defaultdict
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union, cast

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int, Vec3IntLike


@attr.frozen
class BoundingBox:
    """
    This class is used to represent an axis-aligned cuboid in 3D.
    The top-left coordinate is inclusive and the bottom-right coordinate is exclusive.

    A small usage example:

    ```python
    from webknossos import BoundingBox

    bbox_1 = BoundingBox((0, 0, 0), (100, 100, 100))
    bbox_2 = BoundingBox((75, 75, 75), (100, 100, 100))

    assert bbox_1.intersected_with(bbox_2).size == (25, 25, 25)
    ```
    """

    topleft: Vec3Int = attr.field(converter=Vec3Int)
    size: Vec3Int = attr.field(converter=Vec3Int)
    bottomright: Vec3Int = attr.field(init=False)
    name: Optional[str] = "Unnamed Bounding Box"
    is_visible: bool = True
    id: Optional[str] = None
    color: Optional[Tuple[float, float, float, float]] = None

    def __attrs_post_init__(self) -> None:
        if not self.size.is_positive():
            # Flip the size in negative dimensions, so that the topleft is smaller than bottomright.
            # E.g. BoundingBox((10, 10, 10), (-5, 5, 5)) -> BoundingBox((5, 10, 10), (5, 5, 5)).
            negative_size = self.size.pairmin(Vec3Int.zeros())
            new_topleft = self.topleft + negative_size
            new_size = self.size.pairmax(-self.size)
            object.__setattr__(self, "topleft", new_topleft)
            object.__setattr__(self, "size", new_size)

        # Compute bottomright to avoid that it's recomputed every time
        # it is needed.
        object.__setattr__(self, "bottomright", self.topleft + self.size)

    def with_topleft(self, new_topleft: Vec3IntLike) -> "BoundingBox":

        return BoundingBox(new_topleft, self.size)

    def with_size(self, new_size: Vec3IntLike) -> "BoundingBox":

        return BoundingBox(self.topleft, new_size)

    def with_bounds_x(
        self, new_topleft_x: Optional[int] = None, new_size_x: Optional[int] = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.x optionally replaced and size.x optionally replaced."""

        new_topleft = (
            self.topleft.with_x(new_topleft_x)
            if new_topleft_x is not None
            else self.topleft
        )
        new_size = self.size.with_x(new_size_x) if new_size_x is not None else self.size
        return BoundingBox(new_topleft, new_size)

    def with_bounds_y(
        self, new_topleft_y: Optional[int] = None, new_size_y: Optional[int] = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.y optionally replaced and size.y optionally replaced."""

        new_topleft = (
            self.topleft.with_y(new_topleft_y)
            if new_topleft_y is not None
            else self.topleft
        )
        new_size = self.size.with_y(new_size_y) if new_size_y is not None else self.size
        return BoundingBox(new_topleft, new_size)

    def with_bounds_z(
        self, new_topleft_z: Optional[int] = None, new_size_z: Optional[int] = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.z optionally replaced and size.z optionally replaced."""

        new_topleft = (
            self.topleft.with_z(new_topleft_z)
            if new_topleft_z is not None
            else self.topleft
        )
        new_size = self.size.with_z(new_size_z) if new_size_z is not None else self.size
        return BoundingBox(new_topleft, new_size)

    @staticmethod
    def from_wkw_dict(bbox: Dict) -> "BoundingBox":
        return BoundingBox(
            bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]]
        )

    @staticmethod
    def from_config_dict(bbox: Dict) -> "BoundingBox":
        return BoundingBox(bbox["topleft"], bbox["size"])

    @staticmethod
    def from_tuple6(tuple6: Tuple[int, int, int, int, int, int]) -> "BoundingBox":
        return BoundingBox(tuple6[0:3], tuple6[3:6])

    @staticmethod
    def from_tuple2(tuple2: Tuple[Vec3IntLike, Vec3IntLike]) -> "BoundingBox":
        return BoundingBox(tuple2[0], tuple2[1])

    @staticmethod
    def from_points(points: Iterable[Vec3IntLike]) -> "BoundingBox":
        """Returns a bounding box exactly containing all points."""

        all_points = np.array([Vec3Int(point).to_list() for point in points])
        topleft = all_points.min(axis=0)
        bottomright = all_points.max(axis=0)

        # bottomright is exclusive
        bottomright += 1

        return BoundingBox(topleft, bottomright - topleft)

    @staticmethod
    def from_checkpoint_name(checkpoint_name: str) -> "BoundingBox":
        """This function extracts a bounding box in the format `x_y_z_sx_sy_xz` which is contained in a string."""
        regex = r"(([0-9]+_){5}([0-9]+))"
        match = re.search(regex, checkpoint_name)
        assert (
            match is not None
        ), f"Could not extract bounding box from {checkpoint_name}"
        bbox_tuple = tuple(int(value) for value in match.group().split("_"))
        return BoundingBox.from_tuple6(
            cast(Tuple[int, int, int, int, int, int], bbox_tuple)
        )

    @staticmethod
    def from_csv(csv_bbox: str) -> "BoundingBox":
        bbox_tuple = tuple(int(x) for x in csv_bbox.split(","))
        return BoundingBox.from_tuple6(
            cast(Tuple[int, int, int, int, int, int], bbox_tuple)
        )

    @staticmethod
    def from_auto(obj: Union["BoundingBox", str, Dict, List, Tuple]) -> "BoundingBox":
        if isinstance(obj, BoundingBox):
            return obj
        elif isinstance(obj, str):
            if ":" in obj:
                return BoundingBox.from_auto(json.loads(obj))
            else:
                return BoundingBox.from_csv(obj)
        elif isinstance(obj, dict):
            if "size" in obj:
                return BoundingBox.from_config_dict(obj)
            return BoundingBox.from_wkw_dict(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 2:
                return BoundingBox.from_tuple2(obj)  # type: ignore
            elif len(obj) == 6:
                return BoundingBox.from_tuple6(obj)  # type: ignore

        raise Exception("Unknown bounding box format.")

    @staticmethod
    def group_boxes_with_aligned_mag(
        bounding_boxes: Iterable["BoundingBox"], aligning_mag: Mag
    ) -> Dict["BoundingBox", List["BoundingBox"]]:
        """
        Groups the given BoundingBox instances by aligning each
        bbox to the given mag and using that as the key.
        For example, bounding boxes of size 256**3 could be grouped
        into the corresponding 1024**3 chunks to which they belong
        by using aligning_mag = Mag(1024).
        """

        chunks_with_bboxes = defaultdict(list)
        for bbox in bounding_boxes:
            chunk_key = bbox.align_with_mag(aligning_mag, ceil=True)
            chunks_with_bboxes[chunk_key].append(bbox)

        return chunks_with_bboxes

    def to_wkw_dict(self) -> dict:

        (  # pylint: disable=unbalanced-tuple-unpacking
            width,
            height,
            depth,
        ) = self.size.to_list()

        return {
            "topLeft": self.topleft.to_list(),
            "width": width,
            "height": height,
            "depth": depth,
        }

    def to_config_dict(self) -> dict:

        return {"topleft": self.topleft.to_list(), "size": self.size.to_list()}

    def to_checkpoint_name(self) -> str:

        x, y, z = self.topleft
        width, height, depth = self.size
        return "{x}_{y}_{z}_{width}_{height}_{depth}".format(
            x=x, y=y, z=z, width=width, height=height, depth=depth
        )

    def to_tuple6(self) -> Tuple[int, int, int, int, int, int]:

        return tuple(self.topleft.to_list() + self.size.to_list())  # type: ignore

    def to_csv(self) -> str:

        return ",".join(map(str, self.to_tuple6()))

    def __repr__(self) -> str:

        return "BoundingBox(topleft={}, size={})".format(
            str(tuple(self.topleft)), str(tuple(self.size))
        )

    def __str__(self) -> str:

        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoundingBox):
            return self.topleft == other.topleft and self.size == other.size
        else:
            raise NotImplementedError()

    def padded_with_margins(
        self, margins_left: Vec3IntLike, margins_right: Optional[Vec3IntLike] = None
    ) -> "BoundingBox":

        if margins_right is None:
            margins_right = margins_left

        margins_left = Vec3Int(margins_left)
        margins_right = Vec3Int(margins_right)

        return BoundingBox(
            topleft=self.topleft - margins_left,
            size=self.size + (margins_left + margins_right),
        )

    def intersected_with(
        self, other: "BoundingBox", dont_assert: bool = False
    ) -> "BoundingBox":
        """If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0))"""

        topleft = self.topleft.pairmax(other.topleft)
        bottomright = self.bottomright.pairmin(other.bottomright)
        size = (bottomright - topleft).pairmax(Vec3Int.zeros())

        intersection = BoundingBox(topleft, size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self, other: "BoundingBox") -> "BoundingBox":
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        topleft = self.topleft.pairmin(other.topleft)
        bottomright = self.bottomright.pairmax(other.bottomright)
        size = bottomright - topleft

        return BoundingBox(topleft, size)

    def is_empty(self) -> bool:
        return not self.size.is_positive(strictly_positive=True)

    def in_mag(self, mag: Mag) -> "BoundingBox":
        mag_vec = mag.to_vec3_int()

        assert (
            self.topleft % mag_vec == Vec3Int.zeros()
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            self.bottomright % mag_vec == Vec3Int.zeros()
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return BoundingBox(
            topleft=(self.topleft // mag_vec),
            size=(self.size // mag_vec),
        )

    def from_mag_to_mag1(self, from_mag: Mag) -> "BoundingBox":
        mag_vec = from_mag.to_vec3_int()
        return BoundingBox(
            topleft=(self.topleft * mag_vec),
            size=(self.size * mag_vec),
        )

    def _align_with_mag_slow(self, mag: Mag, ceil: bool = False) -> "BoundingBox":
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """
        np_mag = mag.to_np()

        align = (
            lambda point, round_fn: round_fn(point.to_np() / np_mag).astype(int)
            * np_mag
        )

        if ceil:
            topleft = align(self.topleft, np.floor)
            bottomright = align(self.bottomright, np.ceil)
        else:
            topleft = align(self.topleft, np.ceil)
            bottomright = align(self.bottomright, np.floor)
        return BoundingBox(topleft, bottomright - topleft)

    def align_with_mag(
        self, mag: Union[Mag, Vec3Int], ceil: bool = False
    ) -> "BoundingBox":
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """
        # This does the same as _align_with_mag_slow, which is more readable.
        # Same behavior is asserted in test_align_with_mag_against_numpy_implementation
        mag_vec = mag.to_vec3_int() if isinstance(mag, Mag) else mag
        roundup = self.topleft if ceil else self.bottomright
        rounddown = self.bottomright if ceil else self.topleft
        margin_to_roundup = roundup % mag_vec
        aligned_roundup = roundup - margin_to_roundup
        margin_to_rounddown = (mag_vec - (rounddown % mag_vec)) % mag_vec
        aligned_rounddown = rounddown + margin_to_rounddown
        if ceil:
            return BoundingBox(aligned_roundup, aligned_rounddown - aligned_roundup)
        else:
            return BoundingBox(aligned_rounddown, aligned_roundup - aligned_rounddown)

    def contains(self, coord: Union[Vec3IntLike, np.ndarray]) -> bool:
        """Check whether a point is inside of the bounding box.
        Note that the point may have float coordinates in the ndarray case"""

        if isinstance(coord, np.ndarray):
            assert coord.shape == (
                3,
            ), f"Numpy array BoundingBox.contains must have shape (3,), got {coord.shape}."
            return cast(
                bool,
                np.all(coord >= self.topleft) and np.all(coord < self.bottomright),
            )
        else:
            # In earlier versions, we simply converted to ndarray to have
            # a unified calculation here, but this turned out to be a performance bottleneck.
            # Therefore, the contains-check is performed on the tuple here.
            coord = Vec3Int(coord)
            return (
                self.topleft[0] <= coord[0] < self.bottomright[0]
                and self.topleft[1] <= coord[1] < self.bottomright[1]
                and self.topleft[2] <= coord[2] < self.bottomright[2]
            )

    def contains_bbox(self, inner_bbox: "BoundingBox") -> bool:
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self,
        chunk_size: Vec3IntLike,
        chunk_border_alignments: Optional[Vec3IntLike] = None,
    ) -> Generator["BoundingBox", None, None]:
        """Decompose the bounding box into smaller chunks of size `chunk_size`.

        Chunks at the border of the bounding box might be smaller than chunk_size.
        If `chunk_border_alignment` is set, all border coordinates
        *between two chunks* will be divisible by that value.
        """

        start = self.topleft.to_np()
        chunk_size = Vec3Int(chunk_size).to_np()

        start_adjust = np.array([0, 0, 0])
        if chunk_border_alignments is not None:
            chunk_border_alignments_array = Vec3Int(chunk_border_alignments).to_np()
            assert np.all(
                chunk_size % chunk_border_alignments_array == 0
            ), f"{chunk_size} not divisible by {chunk_border_alignments_array}"

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments_array

        for x in range(
            start[0] - start_adjust[0], start[0] + self.size[0], chunk_size[0]
        ):
            for y in range(
                start[1] - start_adjust[1], start[1] + self.size[1], chunk_size[1]
            ):
                for z in range(
                    start[2] - start_adjust[2], start[2] + self.size[2], chunk_size[2]
                ):
                    yield BoundingBox([x, y, z], chunk_size).intersected_with(self)

    def volume(self) -> int:

        return self.size.prod()

    def slice_array(self, array: np.ndarray) -> np.ndarray:

        return array[
            self.topleft.x : self.bottomright.x,
            self.topleft.y : self.bottomright.y,
            self.topleft.z : self.bottomright.z,
        ]

    def to_slices(self) -> Tuple[slice, slice, slice]:
        return np.index_exp[
            self.topleft.x : self.bottomright.x,
            self.topleft.y : self.bottomright.y,
            self.topleft.z : self.bottomright.z,
        ]

    def offset(self, vector: Vec3IntLike) -> "BoundingBox":

        return BoundingBox(self.topleft + Vec3Int(vector), self.size)

    def __hash__(self) -> int:
        return hash(self.to_tuple6())

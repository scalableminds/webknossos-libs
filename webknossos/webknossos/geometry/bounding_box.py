import json
import re
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int, Vec3IntLike


class BoundingBoxNamedTuple(NamedTuple):
    topleft: Tuple[int, int, int]
    size: Tuple[int, int, int]


@attr.frozen
class BoundingBox:
    topleft: Vec3Int = attr.ib(converter=Vec3Int)
    size: Vec3Int = attr.ib(converter=Vec3Int)

    @property
    def bottomright(self) -> Vec3Int:

        return self.topleft + self.size

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
    def from_named_tuple(bb_named_tuple: BoundingBoxNamedTuple) -> "BoundingBox":
        return BoundingBox(bb_named_tuple.topleft, bb_named_tuple.size)

    @staticmethod
    def from_checkpoint_name(checkpoint_name: str) -> "BoundingBox":
        """This function extracts a bounding box in the format x_y_z_sx_sy_xz which is contained in a string."""
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
    def from_auto(
        obj: Union["BoundingBox", str, Dict, BoundingBoxNamedTuple, List, Tuple]
    ) -> "BoundingBox":
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
        elif isinstance(obj, BoundingBoxNamedTuple):
            return BoundingBox.from_named_tuple(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 2:
                return BoundingBox.from_tuple2(obj)  # type: ignore
            elif len(obj) == 6:
                return BoundingBox.from_tuple6(obj)  # type: ignore

        raise Exception("Unknown bounding box format.")

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

    def to_named_tuple(self) -> BoundingBoxNamedTuple:
        return BoundingBoxNamedTuple(
            topleft=cast(Tuple[int, int, int], tuple(self.topleft)),
            size=cast(Tuple[int, int, int], tuple(self.size)),
        )

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

        topleft = np.maximum(self.topleft.to_np(), other.topleft.to_np())
        bottomright = np.minimum(self.bottomright.to_np(), other.bottomright.to_np())
        size = np.maximum(bottomright - topleft, (0, 0, 0))

        intersection = BoundingBox(topleft, size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self, other: "BoundingBox") -> "BoundingBox":

        topleft = np.minimum(self.topleft, other.topleft)
        bottomright = np.maximum(self.bottomright, other.bottomright)
        size = bottomright - topleft

        return BoundingBox(topleft, size)

    def is_empty(self) -> bool:

        return not all(self.size.to_np() > 0)

    def in_mag(self, mag: Mag) -> "BoundingBox":

        np_mag = np.array(mag.to_list())

        assert (
            np.count_nonzero(self.topleft.to_np() % np_mag) == 0
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            np.count_nonzero(self.bottomright.to_np() % np_mag) == 0
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return BoundingBox(
            topleft=(self.topleft // np_mag),
            size=(self.size // np_mag),
        )

    def align_with_mag(self, mag: Mag, ceil: bool = False) -> "BoundingBox":
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """

        np_mag = np.array(mag.to_list())

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

    def contains(self, coord: Vec3IntLike) -> bool:

        coord = Vec3Int(coord).to_np()

        return cast(
            bool,
            np.all(coord >= self.topleft) and np.all(coord < self.topleft + self.size),
        )

    def contains_bbox(self, inner_bbox: "BoundingBox") -> bool:
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self,
        chunk_size: Vec3IntLike,
        chunk_border_alignments: Optional[List[int]] = None,
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

            chunk_border_alignments_array = np.array(chunk_border_alignments)
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

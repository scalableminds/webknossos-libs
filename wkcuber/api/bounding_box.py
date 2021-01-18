# mypy: allow-untyped-defs
import json
import re
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    NamedTuple,
    cast,
)

import numpy as np

from wkcuber.mag import Mag

Shape3D = Union[List[int], Tuple[int, int, int], np.ndarray]


class BoundingBoxNamedTuple(NamedTuple):
    topleft: Tuple[int, int, int]
    size: Tuple[int, int, int]


class BoundingBox:
    def __init__(self, topleft: Shape3D, size: Shape3D):

        self.topleft = np.array(topleft, dtype=np.int)
        self.size = np.array(size, dtype=np.int)

    @property
    def bottomright(self) -> np.ndarray:

        return self.topleft + self.size

    @staticmethod
    def from_wkw(bbox: Dict) -> "BoundingBox":
        return BoundingBox(
            bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]]
        )

    @staticmethod
    def from_config(bbox: Dict) -> "BoundingBox":
        return BoundingBox(bbox["topleft"], bbox["size"])

    @staticmethod
    def from_tuple6(tuple6: Tuple[int, int, int, int, int, int]) -> "BoundingBox":
        return BoundingBox(tuple6[0:3], tuple6[3:6])

    @staticmethod
    def from_tuple2(tuple2: Tuple[Shape3D, Shape3D]) -> "BoundingBox":
        return BoundingBox(tuple2[0], tuple2[1])

    @staticmethod
    def from_points(points: Iterable[Shape3D]) -> "BoundingBox":

        all_points = np.array(points)
        topleft = all_points.min(axis=0)
        bottomright = all_points.max(axis=0)

        # bottomright is exclusive
        bottomright += 1

        return BoundingBox(topleft, bottomright - topleft)

    @staticmethod
    def from_named_tuple(bb_named_tuple: BoundingBoxNamedTuple):

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
        topleft = cast(Tuple[int, int, int], bbox_tuple[:3])
        size = cast(Tuple[int, int, int], bbox_tuple[3:6])
        return BoundingBox.from_tuple2((topleft, size))

    @staticmethod
    def from_csv(csv_bbox: str) -> "BoundingBox":
        bbox_tuple = tuple(int(x) for x in csv_bbox.split(","))
        return BoundingBox.from_tuple6(
            cast(Tuple[int, int, int, int, int, int], bbox_tuple)
        )

    @staticmethod
    def from_auto(obj) -> "BoundingBox":
        if isinstance(obj, BoundingBox):
            return obj
        elif isinstance(obj, str):
            if ":" in obj:
                return BoundingBox.from_auto(json.loads(obj))
            else:
                return BoundingBox.from_csv(obj)
        elif isinstance(obj, dict):
            return BoundingBox.from_wkw(obj)
        elif isinstance(obj, BoundingBoxNamedTuple):
            return BoundingBox.from_named_tuple(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 2:
                return BoundingBox.from_tuple2(obj)  # type: ignore
            elif len(obj) == 6:
                return BoundingBox.from_tuple6(obj)  # type: ignore

        raise Exception("Unknown bounding box format.")

    def as_wkw(self) -> dict:

        width, height, depth = self.size.tolist()

        return {
            "topLeft": self.topleft.tolist(),
            "width": width,
            "height": height,
            "depth": depth,
        }

    def as_config(self) -> dict:

        return {"topleft": self.topleft.tolist(), "size": self.size.tolist()}

    def as_checkpoint_name(self) -> str:

        x, y, z = self.topleft
        width, height, depth = self.size
        return "{x}_{y}_{z}_{width}_{height}_{depth}".format(
            x=x, y=y, z=z, width=width, height=height, depth=depth
        )

    def as_tuple6(self) -> Tuple[int, int, int, int, int, int]:

        return tuple(self.topleft.tolist() + self.size.tolist())  # type: ignore

    def as_csv(self) -> str:

        return ",".join(map(str, self.as_tuple6()))

    def __repr__(self) -> str:

        return "BoundingBox(topleft={}, size={})".format(
            str(tuple(self.topleft)), str(tuple(self.size))
        )

    def __str__(self) -> str:

        return self.__repr__()

    def __eq__(self, other) -> bool:

        return np.array_equal(self.topleft, other.topleft) and np.array_equal(
            self.size, other.size
        )

    def padded_with_margins(
        self, margins_left: Shape3D, margins_right: Optional[Shape3D] = None
    ) -> "BoundingBox":

        if margins_right is None:
            margins_right = margins_left

        margins_left = np.array(margins_left)
        margins_right = np.array(margins_right)

        return BoundingBox(
            topleft=self.topleft - margins_left,
            size=self.size + (margins_left + margins_right),
        )

    def intersected_with(
        self, other: "BoundingBox", dont_assert=False
    ) -> "BoundingBox":
        """ If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0)) """

        topleft = np.maximum(self.topleft, other.topleft)
        bottomright = np.minimum(self.bottomright, other.bottomright)
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

        return not all(self.size > 0)

    def in_mag(self, mag: Mag) -> "BoundingBox":

        np_mag = np.array(mag.to_array())

        assert (
            np.count_nonzero(self.topleft % np_mag) == 0
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            np.count_nonzero(self.bottomright % np_mag) == 0
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return BoundingBox(
            topleft=(self.topleft // np_mag).astype(np.int),
            size=(self.size // np_mag).astype(np.int),
        )

    def align_with_mag(self, mag: Mag, ceil=False):
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """

        np_mag = np.array(mag.to_array())

        align = lambda point, round_fn: round_fn(point / np_mag).astype(np.int) * np_mag

        if ceil:
            topleft = align(self.topleft, np.floor)
            bottomright = align(self.bottomright, np.ceil)
        else:
            topleft = align(self.topleft, np.ceil)
            bottomright = align(self.bottomright, np.floor)
        return BoundingBox(topleft, bottomright - topleft)

    def contains(self, coord: Shape3D) -> bool:

        coord = np.array(coord)

        return np.all(coord >= self.topleft) and np.all(
            coord < self.topleft + self.size
        )

    def contains_bbox(self, inner_bbox: "BoundingBox") -> bool:
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self, chunk_size: Shape3D, chunk_border_alignments: Optional[List[int]] = None
    ) -> Generator["BoundingBox", None, None]:
        """Decompose the bounding box into smaller chunks of size `chunk_size`.

        Chunks at the border of the bounding box might be smaller than chunk_size.
        If `chunk_border_alignment` is set, all border coordinates
        *between two chunks* will be divisible by that value.
        """

        start = self.topleft.copy()
        chunk_size = np.array(chunk_size)

        start_adjust = np.array([0, 0, 0])
        if chunk_border_alignments is not None:

            chunk_border_alignments = np.array(chunk_border_alignments)
            assert np.all(
                chunk_size % chunk_border_alignments == 0
            ), f"{chunk_size} not divisible by {chunk_border_alignments}"

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments

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
            self.topleft[0] : self.bottomright[0],
            self.topleft[1] : self.bottomright[1],
            self.topleft[2] : self.bottomright[2],
        ]

    def as_slices(self) -> Tuple[slice, slice, slice]:
        return np.index_exp[
            self.topleft[0] : self.bottomright[0],
            self.topleft[1] : self.bottomright[1],
            self.topleft[2] : self.bottomright[2],
        ]

    def copy(self) -> "BoundingBox":

        return BoundingBox(self.topleft.copy(), self.bottomright.copy())

    def offset(self, vector: Tuple[int, int, int]) -> "BoundingBox":

        return BoundingBox(self.topleft + np.array(vector), self.size.copy())

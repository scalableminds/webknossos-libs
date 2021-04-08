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
    cast, TypeVar, Generic, Any, overload,
)

import numpy as np

from wkcuber.SafeBoundingBox import AbstractVec, VecNm, VecMag, VecMag1, VecKnownMag
from wkcuber.mag import Mag

Shape3D = Union[List[int], Tuple[int, int, int], np.ndarray]


class BoundingBoxNamedTuple(NamedTuple):
    topleft: Tuple[int, int, int]
    size: Tuple[int, int, int]


VecT = TypeVar("VecT", bound=AbstractVec)


class GenericBoundingBox(Generic[VecT]):
    topleft: VecT
    size: VecT

    def __init__(self, topleft: VecT, size: VecT):
        self.topleft: VecT = topleft
        self.size: VecT = size


    @property
    def bottomright(self) -> VecT:
        return self.topleft + self.size

    @staticmethod
    def from_wkw(bbox: Dict) -> "GenericBoundingBox":
        return GenericBoundingBox(
            bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]]
        )

    @staticmethod
    def from_config(bbox: Dict) -> "GenericBoundingBox":
        return GenericBoundingBox(bbox["topleft"], bbox["size"])

    @staticmethod
    def from_tuple6(tuple6: Tuple[int, int, int, int, int, int]) -> "GenericBoundingBox":
        return GenericBoundingBox(tuple6[0:3], tuple6[3:6])

    @staticmethod
    def from_tuple2(tuple2: Tuple[Shape3D, Shape3D]) -> "GenericBoundingBox":
        return GenericBoundingBox(tuple2[0], tuple2[1])

    @staticmethod
    def from_points(points: Iterable[Shape3D]) -> "GenericBoundingBox":

        all_points = np.array(points)
        topleft = all_points.min(axis=0)
        bottomright = all_points.max(axis=0)

        # bottomright is exclusive
        bottomright += 1

        return GenericBoundingBox(topleft, bottomright - topleft)

    @staticmethod
    def from_named_tuple(bb_named_tuple: BoundingBoxNamedTuple):

        return GenericBoundingBox(bb_named_tuple.topleft, bb_named_tuple.size)

    @staticmethod
    def from_checkpoint_name(checkpoint_name: str) -> "GenericBoundingBox":
        """This function extracts a bounding box in the format x_y_z_sx_sy_xz which is contained in a string."""
        regex = r"(([0-9]+_){5}([0-9]+))"
        match = re.search(regex, checkpoint_name)
        assert (
            match is not None
        ), f"Could not extract bounding box from {checkpoint_name}"
        bbox_tuple = tuple(int(value) for value in match.group().split("_"))
        topleft = cast(Tuple[int, int, int], bbox_tuple[:3])
        size = cast(Tuple[int, int, int], bbox_tuple[3:6])
        return GenericBoundingBox.from_tuple2((topleft, size))

    @staticmethod
    def from_csv(csv_bbox: str) -> "GenericBoundingBox":
        bbox_tuple = tuple(int(x) for x in csv_bbox.split(","))
        return GenericBoundingBox.from_tuple6(
            cast(Tuple[int, int, int, int, int, int], bbox_tuple)
        )

    @staticmethod
    def from_auto(obj) -> "GenericBoundingBox":
        if isinstance(obj, GenericBoundingBox):
            return obj
        elif isinstance(obj, str):
            if ":" in obj:
                return GenericBoundingBox.from_auto(json.loads(obj))
            else:
                return GenericBoundingBox.from_csv(obj)
        elif isinstance(obj, dict):
            return GenericBoundingBox.from_wkw(obj)
        elif isinstance(obj, GenericBoundingBox):
            return GenericBoundingBox.from_named_tuple(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 2:
                return GenericBoundingBox.from_tuple2(obj)  # type: ignore
            elif len(obj) == 6:
                return GenericBoundingBox.from_tuple6(obj)  # type: ignore

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
    ) -> "GenericBoundingBox":

        if margins_right is None:
            margins_right = margins_left

        margins_left = np.array(margins_left)
        margins_right = np.array(margins_right)

        return GenericBoundingBox(
            topleft=self.topleft - margins_left,
            size=self.size + (margins_left + margins_right),
        )

    def intersected_with(
        self, other: "GenericBoundingBox", dont_assert=False
    ) -> "GenericBoundingBox":
        """ If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0)) """

        topleft = np.maximum(self.topleft, other.topleft)
        bottomright = np.minimum(self.bottomright, other.bottomright)
        size = np.maximum(bottomright - topleft, (0, 0, 0))

        intersection = GenericBoundingBox(topleft, size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self, other: "GenericBoundingBox") -> "GenericBoundingBox":

        topleft = np.minimum(self.topleft, other.topleft)
        bottomright = np.maximum(self.bottomright, other.bottomright)
        size = bottomright - topleft

        return GenericBoundingBox(topleft, size)

    def is_empty(self) -> bool:

        return not all(self.size > 0)

    def in_mag(self, mag: Mag) -> "GenericBoundingBox":

        np_mag = np.array(mag.to_array())

        assert (
            np.count_nonzero(self.topleft.a % np_mag) == 0
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            np.count_nonzero(self.bottomright.a % np_mag) == 0
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return GenericBoundingBox(
            topleft=(self.topleft // np_mag),
            size=(self.size // np_mag),
        )

    def align_with_mag(self, mag: Mag, ceil=False):
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """

        np_mag = np.array(mag.to_array())

        align = lambda point, round_fn: round_fn(point / np_mag).astype(np.int) * np_mag

        if ceil:
            topleft = align(self.topleft.a, np.floor)
            bottomright = align(self.bottomright.a, np.ceil)
        else:
            topleft = align(self.topleft.a, np.ceil)
            bottomright = align(self.bottomright.a, np.floor)
        return GenericBoundingBox(type(self.topleft)(topleft), type(self.topleft)(bottomright - topleft))

    def contains(self, coord: Shape3D) -> bool:

        coord = np.array(coord)

        return np.all(coord >= self.topleft) and np.all(
            coord < self.topleft + self.size
        )

    def contains_bbox(self, inner_bbox: "GenericBoundingBox") -> bool:
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self, chunk_size: Shape3D, chunk_border_alignments: Optional[List[int]] = None
    ) -> Generator["GenericBoundingBox", None, None]:
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

                    yield GenericBoundingBox([x, y, z], chunk_size).intersected_with(self)

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

    def copy(self) -> "GenericBoundingBox":

        return GenericBoundingBox(self.topleft.copy(), self.bottomright.copy())

    def offset(self, vector: Tuple[int, int, int]) -> "GenericBoundingBox":

        return GenericBoundingBox(self.topleft + np.array(vector), self.size.copy())


class BoundingBoxNm(GenericBoundingBox[VecNm]):
    @overload
    def as_bb_mag(self, scale: Any) -> "BoundingBoxMag1": ...

    @overload
    def as_bb_mag(self, scale: Any, mag: Mag, ceil: bool = False) -> "BoundingBoxKnownMag": ...

    def as_bb_mag(self, scale: Any, mag: Optional[Any] = None, ceil: bool = False) -> "BoundingBoxKnownMag":
        mag = Mag(mag)
        if mag is None:
            if ceil:
                topleft1 = self.topleft.as_mag(scale)
                bottomright1 = self.bottomright.as_mag(scale)
            else:
                topleft1 = self.topleft.as_mag(scale)
                bottomright1 = self.bottomright.as_mag(scale)
            return BoundingBoxMag1(topleft1, bottomright1 - topleft1)
        else:
            if ceil:
                topleft2 = self.topleft.as_mag(scale, mag, not ceil)
                bottomright2 = self.bottomright.as_mag(scale, mag, ceil)
            else:
                topleft2 = self.topleft.as_mag(scale, mag, ceil)
                bottomright2 = self.bottomright.as_mag(scale, mag, not ceil)
            return BoundingBoxKnownMag(topleft2, bottomright2 - topleft2)


class BoundingBox(GenericBoundingBox[VecMag]):
    def with_mag(self, mag: Mag) -> "BoundingBoxKnownMag":
        return BoundingBoxKnownMag(self.topleft.with_mag(mag), self.size.with_mag(mag))

    def as_nm(self, scale: Any) -> "BoundingBoxNm":
        return BoundingBoxNm(self.topleft.as_nm(scale), self.size.as_nm(scale))


class BoundingBoxKnownMag(BoundingBox):
    top_left: VecKnownMag
    size: VecKnownMag

    def to_mag(self, target_mag: Any, ceil: bool = True) -> "BoundingBoxKnownMag":
        target_mag = Mag(target_mag)
        if ceil:
            topleft = self.topleft.to_mag(target_mag, not ceil)
            bottomright = self.bottomright.to_mag(target_mag, ceil)
        else:
            topleft = self.topleft.to_mag(target_mag, ceil)
            bottomright = self.bottomright.to_mag(target_mag, not ceil)

        return BoundingBoxKnownMag(topleft, bottomright - topleft)


class BoundingBoxMag1(BoundingBoxKnownMag):
    topleft: VecMag1
    size: VecMag1


if __name__ == "__main__":
    # Types of Points:
    # VecMag
    VecMag((10, 20, 40))
    # VecKnownMag
    VecKnownMag((10, 20, 40), Mag([2, 2, 1]))
    # VecMag1
    VecMag1((10, 20, 40))
    # VecNm
    VecNm((10.1, 20.2, 40.4))

    # Types of BoundingBoxes:
    #   BoundingBox
    #   BoundingBoxKnownMag
    #   BoundingBoxMag1
    #   BoundingBoxNm

    # Bounding box without mag
    top_left = VecMag((10, 20, 40))
    size = VecMag((200, 300, 400))
    any_bb = BoundingBox(topleft=top_left, size=size)

    # Bounding box with mag
    top_left = VecMag1((40, 80, 160))
    size = VecMag1((800, 1200, 1600))
    mag1_bb = BoundingBoxMag1(topleft=top_left, size=size)
    mag2_bb = mag1_bb.to_mag(2)
    assert tuple(mag2_bb.topleft.a) == (20, 40, 80)
    assert tuple(mag2_bb.size.a) == (400, 600, 800)

    top_left = VecKnownMag((10, 20, 40), mag=Mag(4))
    size = VecKnownMag((200, 300, 400), mag=Mag(4))
    mag4_bb = BoundingBoxKnownMag(topleft=top_left, size=size)
    mag8_bb = mag4_bb.to_mag(8)
    assert tuple(mag8_bb.topleft.a) == (5, 10, 20)
    assert tuple(mag8_bb.size.a) == (100, 150, 200)
    mag_nm_from_8 = mag8_bb.as_nm(scale=(4.4, 4.4, 2.2))
    assert tuple(mag_nm_from_8.topleft.a) == (176, 352, 352)
    assert tuple(mag_nm_from_8.size.a) == (3520, 5280, 3520)

    # Bounding box with nm
    top_left = VecNm((100.1, 200.2, 400.4))
    size = VecNm((200, 400, 800))
    nm_bb = BoundingBoxNm(topleft=top_left, size=size)
    mag2_form_nm = nm_bb.as_bb_mag(scale=(2, 2, 1), mag=Mag(2))
    assert tuple(mag2_form_nm.topleft.a) == (25, 50, 200)
    assert tuple(mag2_form_nm.size.a) == (50, 100, 400)


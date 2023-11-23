import re
from collections import defaultdict
from itertools import product
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union, cast

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int
from .vecn_int import VecNInt, VecNIntLike

_DEFAULT_BBOX_NAME = "Unnamed Bounding Box"

@attr.frozen
class NDBoundingBox:
    """
    This class is used to represent an axis-aligned cuboid in N-D.
    The top-left coordinate is inclusive and the bottom-right coordinate is exclusive.

    A small usage example:

    ```python
    from webknossos import NDBoundingBox

    bbox_1 = NDBoundingBox(top_left=(0, 0, 0), size=(100, 100, 100), axes=("x", "y", "z"))
    bbox_2 = NDBoundingBox(top_left=(75, 75, 75, 0), size=(100, 100, 100, 20), axes=("x", "y", "z", "t"))

    ```
    """

    topleft: VecNInt = attr.field(converter=VecNInt)
    size: VecNInt = attr.field(converter=VecNInt)
    axes: Tuple[str, ...] = attr.field(converter=tuple)
    bottomright: VecNInt = attr.field(init=False)
    name: Optional[str] = _DEFAULT_BBOX_NAME
    is_visible: bool = True
    color: Optional[Tuple[float, float, float, float]] = None

    def __attrs_post_init__(self) -> None:
        assert len(self.topleft) == len(self.size) == len(self.axes), (
            f"The dimensions of topleft, size and axes ({len(self.topleft)}, "
            + f"{len(self.size)} and {len(self.axes)} dimensions) do not match."
        )

        # Bring topleft and size in required order ('x', 'y', 'z', ...)
        try:
            source = [self.axes.index('x'), self.axes.index('y'), self.axes.index('z')]
        except ValueError as err:
            raise ValueError("There are at least 3 dimensions needed with names `x`, `y` and `z`.") from err
        target = [1,2,3]
        self.size.moveaxis(source, target)
        self.topleft.moveaxis(source, target)
        object.__setattr__(self, "axes", ("x", "y", "z", *(e for e in self.axes if e not in ["x", "y", "z"])))

        if not self.size.is_positive():
            # Flip the size in negative dimensions, so that the topleft is smaller than bottomright.
            # E.g. BoundingBox((10, 10, 10), (-5, 5, 5)) -> BoundingBox((5, 10, 10), (5, 5, 5)).
            negative_size = (min(0, value) for value in self.size)
            new_topleft = (val1 + val2 for val1, val2 in zip(self.topleft, negative_size))
            new_size = (max(value, -value) for value in self.size)
            object.__setattr__(self, "topleft", new_topleft)
            object.__setattr__(self, "size", new_size)


        # Compute bottomright to avoid that it's recomputed every time
        # it is needed.
        object.__setattr__(self, "bottomright", self.topleft + self.size)

    def with_additional_axis(self, name: str, extent: Tuple[int, int]) -> "NDBoundingBox":
        assert name not in self.axes, "The identifier of the axis is already taken."
        start, end = extent
        return attr.evolve(
            self,
            topleft=(*self.topleft, start),
            size=(*self.size, end - start),
            axes=(*self.axes, name)
        )
    
    def with_name(self, name: Optional[str]) -> "NDBoundingBox":
        return attr.evolve(self, name=name)

    def with_is_visible(self, is_visible: bool) -> "NDBoundingBox":
        return attr.evolve(self, is_visible=is_visible)

    def with_color(
        self, color: Optional[Tuple[float, float, float, float]]
    ) -> "NDBoundingBox":
        return attr.evolve(self, color=color)
    
    def with_bounds(self, axis: str, new_topleft: Optional[int], new_size: Optional[int]) -> "NDBoundingBox":
        try:
            index = self.axes.index(axis)
        except ValueError as err:
            raise ValueError("The given axis name does not exist.") from err
        
        _new_topleft = (
            self.topleft.with_replaced(index, new_topleft)
            if new_topleft is not None
            else self.topleft
        )
        _new_size = (
            self.size.with_replaced(index, new_size)
            if new_size is not None
            else self.size
        )

        return attr.evolve(self, topleft=_new_topleft, size=_new_size)

    @classmethod
    def from_checkpoint_name(cls, checkpoint_name: str) -> "NDBoundingBox":
        """This function extracts a bounding box in the format `x_y_z_sx_sy_xz` which is contained in a string."""
        regex = r"(([0-9]+_){5}([0-9]+))"
        match = re.search(regex, checkpoint_name)
        assert (
            match is not None
        ), f"Could not extract bounding box from {checkpoint_name}"
        bbox_tuple = tuple(int(value) for value in match.group().split("_"))
        return cls.from_tuple6(cast(Tuple[int, int, int, int, int, int], bbox_tuple))

    @classmethod
    def from_csv(cls, csv_bbox: str) -> "NDBoundingBox":
        bbox_tuple = tuple(int(x) for x in csv_bbox.split(","))
        return cls.from_tuple6(cast(Tuple[int, int, int, int, int, int], bbox_tuple))

    @classmethod
    def from_auto(
        cls, obj: Union["NDBoundingBox", str, Dict, List, Tuple]
    ) -> "NDBoundingBox":
        raise NotImplementedError

    @classmethod
    def group_boxes_with_aligned_mag(
        cls, bounding_boxes: Iterable["NDBoundingBox"], aligning_mag: Mag
    ) -> Dict["NDBoundingBox", List["NDBoundingBox"]]:
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
        ( 
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
        return f"{'_'.join(str(element) for element in self.topleft)}_{'_'.join(str(element) for element in self.size)}"

    def to_csv(self) -> str:
        return ",".join(map(str, self.to_tuple6()))

    def __repr__(self) -> str:
        return f"NDBoundingBox(topleft={self.topleft}, size={self.size}, axes={self.axes})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NDBoundingBox):
            return self.topleft == other.topleft and self.size == other.size
        else:
            raise NotImplementedError()

    def padded_with_margins(
        self, margins_left: VecNIntLike, margins_right: Optional[VecNIntLike] = None
    ) -> None:
        raise NotImplementedError()

    def intersected_with(
        self, other: "NDBoundingBox", dont_assert: bool = False
    ) -> "NDBoundingBox":
        """If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0))"""

        topleft = self.topleft.pairmax(other.topleft)
        bottomright = self.bottomright.pairmin(other.bottomright)
        size = (bottomright - topleft).pairmax(VecNInt.zeros(len(self.size)))

        intersection = attr.evolve(self, topleft=topleft, size=size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self, other: "NDBoundingBox") -> "NDBoundingBox":
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        topleft = self.topleft.pairmin(other.topleft)
        bottomright = self.bottomright.pairmax(other.bottomright)
        size = bottomright - topleft

        return attr.evolve(self, topleft=topleft, size=size)

    def is_empty(self) -> bool:
        return not self.size.is_positive(strictly_positive=True)

    def in_mag(self, mag: Mag) -> "NDBoundingBox":
        mag_vec = mag.to_vec3_int()

        assert (
            Vec3Int(self.topleft.to_xyz()) % mag_vec == Vec3Int.zeros()
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            self.bottomright % mag_vec == Vec3Int.zeros()
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return attr.evolve(
            self,
            topleft=(self.topleft // mag_vec),
            size=(self.size // mag_vec),
        )

    def from_mag_to_mag1(self, from_mag: Mag) -> "NDBoundingBox":
        mag_vec = from_mag.to_vec3_int()
        return attr.evolve(
            self,
            topleft=(self.topleft * mag_vec),
            size=(self.size * mag_vec),
        )

    def _align_with_mag_slow(self, mag: Mag, ceil: bool = False) -> "NDBoundingBox":
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
        return attr.evolve(self, topleft=topleft, size=bottomright - topleft)

    def align_with_mag(
        self, mag: Union[Mag, Vec3Int], ceil: bool = False
    ) -> "NDBoundingBox":
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
            return attr.evolve(
                self, topleft=aligned_roundup, size=aligned_rounddown - aligned_roundup
            )
        else:
            return attr.evolve(
                self,
                topleft=aligned_rounddown,
                size=aligned_roundup - aligned_rounddown,
            )

    def contains(self, coord: VecNIntLike) -> bool:
        """Check whether a point is inside of the bounding box.
        Note that the point may have float coordinates in the ndarray case"""

        if isinstance(coord, np.ndarray):
            assert coord.shape == (
                len(self.size),
            ), f"Numpy array BoundingBox.contains must have shape ({len(self.size)},), got {coord.shape}."
            return cast(
                bool,
                np.all(coord >= self.topleft) and np.all(coord < self.bottomright),
            )
        else:
            # In earlier versions, we simply converted to ndarray to have
            # a unified calculation here, but this turned out to be a performance bottleneck.
            # Therefore, the contains-check is performed on the tuple here.
            coord = VecNInt(coord)
            return all(
                self.topleft[i] <= coord[i] < self.bottomright[i]
                for i in range(len(self.axes))
            )

    def contains_bbox(self, inner_bbox: "NDBoundingBox") -> bool:
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self,
        chunk_shape: VecNIntLike,
        chunk_border_alignments: Optional[VecNIntLike] = None,
    ) -> Generator["NDBoundingBox", None, None]:
        """Decompose the bounding box into smaller chunks of size `chunk_shape`.

        Chunks at the border of the bounding box might be smaller than chunk_shape.
        If `chunk_border_alignment` is set, all border coordinates
        *between two chunks* will be divisible by that value.
        """

        start = self.topleft.to_np()
        chunk_shape = VecNInt(chunk_shape).to_np()

        start_adjust = VecNInt.zeros(len(self.topleft)).to_np()
        if chunk_border_alignments is not None:
            chunk_border_alignments_array = Vec3Int(chunk_border_alignments).to_np()
            assert np.all(
                chunk_shape % chunk_border_alignments_array == 0
            ), f"{chunk_shape} not divisible by {chunk_border_alignments_array}"

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments_array
        for coordinates in product(*[
                range(start[i] - start_adjust[i], start[i] + self.size[i], chunk_shape[i])
                for i in range(len(self.axes))
            ]):

            yield NDBoundingBox(topleft=coordinates, size=chunk_shape, axes=self.axes)

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

    def offset(self, vector: VecNIntLike) -> "NDBoundingBox":
        return attr.evolve(self, topleft=self.topleft + VecNInt(vector))

    def __hash__(self) -> int:
        return hash(self.to_tuple6())

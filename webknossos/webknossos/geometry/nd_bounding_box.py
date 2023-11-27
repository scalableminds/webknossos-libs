from collections import defaultdict
from itertools import product
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union, cast

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int
from .vec_int import VecInt, VecIntLike

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

    topleft: VecInt = attr.field(converter=VecInt)
    size: VecInt = attr.field(converter=VecInt)
    axes: Tuple[str, ...] = attr.field(converter=tuple)
    bottomright: VecInt = attr.field(init=False)
    name: Optional[str] = _DEFAULT_BBOX_NAME
    is_visible: bool = True
    color: Optional[Tuple[float, float, float, float]] = None

    def __attrs_post_init__(self) -> None:
        assert len(self.topleft) == len(self.size) == len(self.axes), (
            f"The dimensions of topleft, size and axes ({len(self.topleft)}, "
            + f"{len(self.size)} and {len(self.axes)} dimensions) do not match."
        )
        if not self._is_sorted():
            self._sort_positions_of_axes()

        if not self.size.is_positive():
            # Flip the size in negative dimensions, so that the topleft is smaller than bottomright.
            # E.g. BoundingBox((10, 10, 10), (-5, 5, 5)) -> BoundingBox((5, 10, 10), (5, 5, 5)).
            negative_size = (min(0, value) for value in self.size)
            new_topleft = (
                val1 + val2 for val1, val2 in zip(self.topleft, negative_size)
            )
            new_size = (max(value, -value) for value in self.size)
            object.__setattr__(self, "topleft", new_topleft)
            object.__setattr__(self, "size", new_size)

        # Compute bottomright to avoid that it's recomputed every time
        # it is needed.
        object.__setattr__(self, "bottomright", self.topleft + self.size)

    def _sort_positions_of_axes(self) -> None:
        # Bring topleft and size in required order
        # ('x', 'y', 'z', <alphabetically sorted remaining axes>)

        size, topleft, axes = zip(
            *sorted(zip(self.size, self.topleft, self.axes), key=lambda x: x[2])
        )
        object.__setattr__(self, "size", VecInt(size))
        object.__setattr__(self, "topleft", VecInt(topleft))
        object.__setattr__(self, "axes", axes)
        try:
            source = [self.axes.index("x"), self.axes.index("y"), self.axes.index("z")]
        except ValueError as err:
            raise ValueError(
                "There are at least 3 dimensions needed with names `x`, `y` and `z`."
            ) from err
        target = [0, 1, 2]
        object.__setattr__(self, "size", self.size.moveaxis(source, target))
        object.__setattr__(self, "topleft", self.topleft.moveaxis(source, target))
        object.__setattr__(
            self,
            "axes",
            ("x", "y", "z", *(e for e in self.axes if e not in ["x", "y", "z"])),
        )

    def _is_sorted(self) -> bool:
        if self.axes[0:3] != ["x", "y", "z"]:
            return False
        return all(
            self.axes[i] < self.axes[i + 1] for i in range(3, len(self.axes) - 2)
        )

    def with_additional_axis(
        self, name: str, extent: Tuple[int, int]
    ) -> "NDBoundingBox":
        assert name not in self.axes, "The identifier of the axis is already taken."
        start, end = extent
        return attr.evolve(
            self,
            topleft=(*self.topleft, start),
            size=(*self.size, end - start),
            axes=(*self.axes, name),
        )

    def with_name(self, name: Optional[str]) -> "NDBoundingBox":
        return attr.evolve(self, name=name)

    def with_is_visible(self, is_visible: bool) -> "NDBoundingBox":
        return attr.evolve(self, is_visible=is_visible)

    def with_color(
        self, color: Optional[Tuple[float, float, float, float]]
    ) -> "NDBoundingBox":
        return attr.evolve(self, color=color)

    def with_bounds(
        self, axis: str, new_topleft: Optional[int], new_size: Optional[int]
    ) -> "NDBoundingBox":
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

    def __repr__(self) -> str:
        return f"NDBoundingBox(topleft={self.topleft.to_tuple()}, size={self.size.to_tuple()}, axes={self.axes})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NDBoundingBox):
            self._check_compatibility(other)
            return self.topleft == other.topleft and self.size == other.size
        else:
            raise NotImplementedError()

    def _check_compatibility(self, other) -> None:
        if self.axes == other.axes:
            return
        else:
            raise ValueError(
                f"Operation with two bboxes is only possible if they have the same axes. {self.axes} != {other.axes}"
            )

    def padded_with_margins(
        self, margins_left: VecIntLike, margins_right: Optional[VecIntLike] = None
    ) -> None:
        raise NotImplementedError()

    def intersected_with(
        self, other: "NDBoundingBox", dont_assert: bool = False
    ) -> "NDBoundingBox":
        """If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0))"""

        self._check_compatibility(other)
        topleft = self.topleft.pairmax(other.topleft)
        bottomright = self.bottomright.pairmin(other.bottomright)
        size = (bottomright - topleft).pairmax(VecInt.zeros(len(self.size)))

        intersection = attr.evolve(self, topleft=topleft, size=size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self, other: "NDBoundingBox") -> "NDBoundingBox":
        self._check_compatibility(other)
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

    def contains(self, coord: VecIntLike) -> bool:
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
            coord = VecInt(coord)
            return all(
                self.topleft[i] <= coord[i] < self.bottomright[i]
                for i in range(len(self.axes))
            )

    def contains_bbox(self, inner_bbox: "NDBoundingBox") -> bool:
        self._check_compatibility(inner_bbox)
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self,
        chunk_shape: VecIntLike,
        chunk_border_alignments: Optional[VecIntLike] = None,
    ) -> Generator["NDBoundingBox", None, None]:
        """Decompose the bounding box into smaller chunks of size `chunk_shape`.

        Chunks at the border of the bounding box might be smaller than chunk_shape.
        If `chunk_border_alignment` is set, all border coordinates
        *between two chunks* will be divisible by that value.
        """

        start = self.topleft.to_np()
        chunk_shape = VecInt(chunk_shape).to_np()

        start_adjust = VecInt.zeros(len(self.topleft)).to_np()
        if chunk_border_alignments is not None:
            chunk_border_alignments_array = Vec3Int(chunk_border_alignments).to_np()
            assert np.all(
                chunk_shape % chunk_border_alignments_array == 0
            ), f"{chunk_shape} not divisible by {chunk_border_alignments_array}"

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments_array
        for coordinates in product(
            *[
                range(
                    start[i] - start_adjust[i], start[i] + self.size[i], chunk_shape[i]
                )
                for i in range(len(self.axes))
            ]
        ):
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

    def offset(self, vector: VecIntLike) -> "NDBoundingBox":
        return attr.evolve(self, topleft=self.topleft + VecInt(vector))

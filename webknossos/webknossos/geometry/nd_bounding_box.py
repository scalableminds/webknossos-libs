from collections import defaultdict
from itertools import product
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union, cast

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int, Vec3IntLike
from .vec_int import VecInt, VecIntLike

_DEFAULT_BBOX_NAME = "Unnamed Bounding Box"


@attr.frozen
class NDBoundingBox:
    """
    This class is used to represent an axis-aligned, n-dimensional cuboid.
    The top-left coordinate is inclusive and the bottom-right coordinate is exclusive.
    The index parameter defines the axis order of the data, all values have to be greater or equal to 1, as index 0 is reserved for channel data.

    A small usage example:

    ```python
    from webknossos import NDBoundingBox

    bbox_1 = NDBoundingBox(top_left=(0, 0, 0), size=(100, 100, 100), axes=("x", "y", "z"), index=(1,2,3))
    bbox_2 = NDBoundingBox(top_left=(75, 75, 75, 0), size=(100, 100, 100, 20), axes=("x", "y", "z", "t"), index=(2,3,4,1))

    ```
    """

    topleft: VecInt = attr.field(converter=VecInt)
    size: VecInt = attr.field(converter=VecInt)
    axes: Tuple[str, ...] = attr.field(converter=tuple)
    index: VecInt = attr.field(converter=VecInt)
    bottomright: VecInt = attr.field(init=False)
    name: Optional[str] = _DEFAULT_BBOX_NAME
    is_visible: bool = True
    color: Optional[Tuple[float, float, float, float]] = None

    def __attrs_post_init__(self) -> None:
        assert (
            len(self.topleft) == len(self.size) == len(self.axes) == len(self.index)
        ), (
            f"The dimensions of topleft, size, axes and index ({len(self.topleft)}, "
            + f"{len(self.size)}, {len(self.axes)} and {len(self.index)}) do not match."
        )
        assert 0 not in self.index, "Index 0 is reserved for channels."
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
        # defined in axisOrder and index of additionalAxes

        size, topleft, axes, index = zip(
            *sorted(
                zip(self.size, self.topleft, self.axes, self.index), key=lambda x: x[3]
            )
        )
        object.__setattr__(self, "size", VecInt(size))
        object.__setattr__(self, "topleft", VecInt(topleft))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "index", index)

    def _is_sorted(self) -> bool:
        return all(self.index[i - 1] < self.index[i] for i in range(1, len(self.index)))

    def with_additional_axis(
        self, name: str, extent: Tuple[int, int], index: Optional[int] = None
    ) -> "NDBoundingBox":
        assert name not in self.axes, "The identifier of the axis is already taken."
        start, end = extent
        return attr.evolve(
            self,
            topleft=(*self.topleft, start),
            size=(*self.size, end - start),
            axes=(*self.axes, name),
            index=(*self.index, index if not index is None else max(self.index) + 1),
        )

    def with_name(self, name: Optional[str]) -> "NDBoundingBox":
        return attr.evolve(self, name=name)

    def with_size(self, new_size: VecIntLike) -> "NDBoundingBox":
        return attr.evolve(self, size=VecInt(new_size))

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

    def get_bounds(self, axis: str) -> Tuple[int, int]:
        try:
            index = self.axes.index(axis)
        except ValueError as err:
            raise ValueError("The given axis name does not exist.") from err

        return (self.topleft[index], self.topleft[index] + self.size[index])

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

    @classmethod
    def from_wkw_dict(cls, bbox: Dict) -> "NDBoundingBox":
        topleft: List[int] = bbox["topLeft"]
        size: List[int] = [bbox["width"], bbox["height"], bbox["depth"]]
        axes: List[str] = ["x", "y", "z"]
        index: List[int] = [1, 2, 3]

        if "axisOrder" in bbox:
            axes = list(bbox["axisOrder"].keys())
            index = [bbox["axisOrder"][axis] for axis in axes]

            if "additionalAxes" in bbox:
                assert (
                    "axisOrder" in bbox
                ), "If there are additionalAxes an axisOrder needs to be provided."
                for axis in bbox["additionalAxes"]:
                    topleft.append(axis["bounds"][0])
                    size.append(axis["bounds"][1] - axis["bounds"][0])
                    axes.append(axis["name"])
                    index.append(axis["index"])

        return cls(topleft, size, axes, index)

    def to_wkw_dict(self) -> dict:
        topleft = [None, None, None]
        width, height, depth = None, None, None
        additional_axes = []
        for i, axis in enumerate(self.axes):
            if axis == "x":
                topleft[0] = self.topleft[i]
                width = self.size[i]
            elif axis == "y":
                topleft[1] = self.topleft[i]
                height = self.size[i]
            elif axis == "z":
                topleft[2] = self.topleft[i]
                depth = self.size[i]
            else:
                additional_axes.append(
                    {
                        "name": axis,
                        "bounds": [self.topleft[i], self.bottomright[i]],
                        "index": self.index[i],
                    }
                )
        if additional_axes:
            return {
                "topLeft": topleft,
                "width": width,
                "height": height,
                "depth": depth,
                "additionalAxes": additional_axes,
            }
        return {
            "topLeft": topleft,
            "width": width,
            "height": height,
            "depth": depth,
        }

    def to_config_dict(self) -> dict:
        return {
            "topleft": self.topleft.to_list(),
            "size": self.size.to_list(),
            "axes": self.axes,
        }

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

        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.axes)

    def get_shape(self, axis_name: str) -> int:
        try:
            index = self.axes.index(axis_name)
            return self.size[index]
        except ValueError as err:
            raise ValueError(
                f"Axis {axis_name} doesn't exist in NDBoundingBox."
            ) from err

    def get_xyz_for_attribute(self, attr_name: str) -> Vec3Int:
        axes = ("x", "y", "z")
        attr_3d = []

        for axis in axes:
            index = self.axes.index(axis)
            attr_3d.append(getattr(self, attr_name)[index])

        return Vec3Int(attr_3d)

    def set_3d(self, attr_name: str, value: Vec3IntLike) -> VecInt:
        value = Vec3Int(value)
        axes = ("x", "y", "z")
        modified_attr = getattr(self, attr_name).to_list()

        for i, axis in enumerate(axes):
            index = self.axes.index(axis)
            modified_attr[index] = value[i]

        return VecInt(modified_attr)

    def _check_compatibility(self, other) -> None:
        """Checks if two bounding boxes are comparable. To be comparable they need the same number of axes, with same names and same order."""
        if self.axes != other.axes:
            raise ValueError(
                f"Operation with two bboxes is only possible if they have the same axes and axes order. {self.axes} != {other.axes}"
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
            self.get_3d("topleft") % mag_vec == Vec3Int.zeros()
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            self.get_3d("bottomright") % mag_vec == Vec3Int.zeros()
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        new_topleft = self.set_3d("topleft", Vec3Int(self.get_3d("topleft") // mag_vec))
        new_size = self.set_3d("size", Vec3Int(self.get_3d("size") // mag_vec))

        return attr.evolve(
            self,
            topleft=new_topleft,
            size=new_size,
        )

    def from_mag_to_mag1(self, from_mag: Mag) -> "NDBoundingBox":
        mag_vec = from_mag.to_vec3_int()
        new_topleft = self.set_3d("topleft", Vec3Int(self.get_3d("topleft") * mag_vec))
        new_size = self.set_3d("size", Vec3Int(self.get_3d("size") * mag_vec))
        return attr.evolve(
            self,
            topleft=new_topleft,
            size=new_size,
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
        topleft = self.get_3d("topleft")
        bottomright = self.get_3d("bottomright")
        roundup = topleft if ceil else bottomright
        rounddown = bottomright if ceil else topleft
        margin_to_roundup = roundup % mag_vec
        aligned_roundup = roundup - margin_to_roundup
        margin_to_rounddown = (mag_vec - (rounddown % mag_vec)) % mag_vec
        aligned_rounddown = rounddown + margin_to_rounddown
        if ceil:
            return attr.evolve(
                self,
                topleft=self.set_3d("topleft", Vec3Int(aligned_roundup)),
                size=self.set_3d("size", Vec3Int(aligned_rounddown - aligned_roundup)),
            )
        else:
            return attr.evolve(
                self,
                topleft=self.set_3d("topleft", Vec3Int(aligned_rounddown)),
                size=self.set_3d("size", Vec3Int(aligned_roundup - aligned_rounddown)),
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
        try:
            # If a 3D chunk_shape is given it is assumed that iteration over xyz is
            # intended. Therefore NDBoundingBoxes are generated that have a shape of
            # x: chunk_shape.x, y: chunk_shape.y, z: chunk_shape.z and 1 for all other
            # axes.
            chunk_shape = Vec3Int(chunk_shape)

            chunk_shape = (
                self.with_size(VecInt.ones(len(self)))
                .set_3d("size", chunk_shape)
                .to_np()
            )
        except AssertionError:
            chunk_shape = VecInt(chunk_shape).to_np()

        start_adjust = VecInt.zeros(len(self)).to_np()
        if chunk_border_alignments is not None:
            try:
                chunk_border_alignments = Vec3Int(chunk_border_alignments)

                chunk_border_alignments = (
                    self.with_size(VecInt.ones(len(self)))
                    .set_3d("size", chunk_border_alignments)
                    .to_np()
                )
            except AssertionError:
                chunk_border_alignments = VecInt(chunk_border_alignments).to_np()

            assert np.all(
                chunk_shape % chunk_border_alignments == 0
            ), f"{chunk_shape} not divisible by {chunk_border_alignments}"

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments
        for coordinates in product(
            *[
                range(
                    start[i] - start_adjust[i], start[i] + self.size[i], chunk_shape[i]
                )
                for i in range(len(self.axes))
            ]
        ):
            yield self.intersected_with(
                NDBoundingBox(
                    topleft=coordinates,
                    size=chunk_shape,
                    axes=self.axes,
                    index=self.index,
                )
            )

    def volume(self) -> int:
        return self.size.prod()

    def slice_array(self, array: np.ndarray) -> np.ndarray:
        return array[self.to_slices()]

    def to_slices(self) -> Tuple[slice, ...]:
        return tuple(
            slice(topleft, topleft + size)
            for topleft, size in zip(self.topleft, self.size)
        )

    def offset(self, vector: VecIntLike) -> "NDBoundingBox":
        vec_int = VecInt(vector)
        if len(vec_int) == 3:
            new_topleft = self.set_3d(
                "topleft", Vec3Int(self.get_3d("topleft") + vec_int)
            )
            return attr.evolve(self, topleft=new_topleft)
        else:
            return attr.evolve(self, topleft=self.topleft + vec_int)

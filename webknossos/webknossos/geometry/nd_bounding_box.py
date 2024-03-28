from collections import defaultdict
from itertools import product
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import attr
import numpy as np

from .mag import Mag
from .vec3_int import Vec3Int, Vec3IntLike
from .vec_int import VecInt, VecIntLike

_DEFAULT_BBOX_NAME = "Unnamed Bounding Box"

_T = TypeVar("_T", bound="NDBoundingBox")


def str_tpl(str_list: Iterable[str]) -> Tuple[str, ...]:
    # Fix for mypy bug https://github.com/python/mypy/issues/5313.
    # Solution based on other issue for the same bug: https://github.com/python/mypy/issues/8389.
    return tuple(str_list)


def int_tpl(vec_int_like: VecIntLike) -> VecInt:
    return VecInt(
        vec_int_like, axes=(f"unset_{i}" for i in range(len(list(vec_int_like))))
    )


@attr.frozen
class NDBoundingBox:
    """
    The NDBoundingBox class is a generalized version of the 3-dimensional BoundingBox class. It is designed to represent bounding boxes in any number of dimensions.

    The bounding box is characterized by its top-left corner, the size of the box, the names of the axes for each dimension, and the index (or order) of the axes. Each axis must have a unique index, starting from 1 (index 0 is reserved for channel information).

    The top-left coordinate is inclusive, while the bottom-right coordinate is exclusive.

    Here's a brief example of how to use it:

    ```python

    # Create a 2D bounding box
    bbox_1 = NDBoundingBox(
        top_left=(0, 0),
        size=(100, 100),
        axes=("x", "y"),
        index=(1,2)
    )

    # Create a 4D bounding box
    bbox_2 = NDBoundingBox(
        top_left=(75, 75, 75, 0),
        size=(100, 100, 100, 20),
        axes=("x", "y", "z", "t"),
        index=(2,3,4,1)
    )
    ```
    """

    topleft: VecInt = attr.field(converter=int_tpl)
    size: VecInt = attr.field(converter=int_tpl)
    axes: Tuple[str, ...] = attr.field(converter=str_tpl)
    index: VecInt = attr.field(converter=int_tpl)
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

        # Convert the delivered tuples to VecInts
        object.__setattr__(self, "topleft", VecInt(self.topleft, axes=self.axes))
        object.__setattr__(self, "size", VecInt(self.size, axes=self.axes))
        object.__setattr__(self, "index", VecInt(self.index, axes=self.axes))

        if not self._is_sorted():
            self._sort_positions_of_axes()

        if not self.size.is_positive():
            # Flip the size in negative dimensions, so that the topleft is smaller than bottomright.
            # E.g. BoundingBox((10, 10, 10), (-5, 5, 5)) -> BoundingBox((5, 10, 10), (5, 5, 5)).
            negative_size = tuple(min(0, value) for value in self.size)
            new_topleft = tuple(
                val1 + val2 for val1, val2 in zip(self.topleft, negative_size)
            )
            new_size = (abs(value) for value in self.size)
            object.__setattr__(self, "topleft", VecInt(new_topleft, axes=self.axes))
            object.__setattr__(self, "size", VecInt(new_size, axes=self.axes))

        # Compute bottomright to avoid that it's recomputed every time
        # it is needed.
        object.__setattr__(
            self,
            "bottomright",
            self.topleft + self.size,
        )

    def _sort_positions_of_axes(self) -> None:
        # Bring topleft and size in required order
        # defined in axisOrder and index of additionalAxes

        size, topleft, axes, index = zip(
            *sorted(
                zip(self.size, self.topleft, self.axes, self.index), key=lambda x: x[3]
            )
        )
        object.__setattr__(self, "size", VecInt(size, axes=axes))
        object.__setattr__(self, "topleft", VecInt(topleft, axes=axes))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "index", VecInt(index, axes=axes))

    def _is_sorted(self) -> bool:
        return all(self.index[i - 1] < self.index[i] for i in range(1, len(self.index)))

    def with_name(self: _T, name: Optional[str]) -> _T:
        """
        Returns a new instance of `NDBoundingBox` with the specified name.

        Args:
        - name (Optional[str]): The name to assign to the new `NDBoundingBox` instance.

        Returns:
        - NDBoundingBox: A new instance of `NDBoundingBox` with the specified name.
        """
        return attr.evolve(self, name=name)

    def with_topleft(self: _T, new_topleft: VecIntLike) -> _T:
        """
        Returns a new NDBoundingBox object with the specified top left coordinates.

        Args:
        - new_topleft (VecIntLike): The new top left coordinates for the bounding box.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated top left coordinates.
        """
        return attr.evolve(self, topleft=VecInt(new_topleft, axes=self.axes))

    def with_size(self: _T, new_size: VecIntLike) -> _T:
        """
        Returns a new NDBoundingBox object with the specified size.

        Args:
        - new_size (VecIntLike): The new size of the bounding box. Can be a VecInt or any object that can be converted to a VecInt.

        Returns:
        - A new NDBoundingBox object with the specified size.
        """
        return attr.evolve(self, size=VecInt(new_size, axes=self.axes))

    def with_index(self: _T, new_index: VecIntLike) -> _T:
        """
        Returns a new NDBoundingBox object with the specified index.

        Args:
        - new_index (VecIntLike): The new axis order for the bounding box.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated index.
        """
        axes, _ = zip(*sorted(zip(self.axes, new_index), key=lambda x: x[1]))
        return attr.evolve(self, index=VecInt(new_index, axes=axes))

    def with_bottomright(self: _T, new_bottomright: VecIntLike) -> _T:
        """
        Returns a new NDBoundingBox with an updated bottomright value.

        Args:
        - new_bottomright (VecIntLike): The new bottom right corner coordinates.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated bottom right corner.
        """
        new_size = VecInt(new_bottomright, axes=self.axes) - self.topleft

        return self.with_size(new_size)

    def with_is_visible(self: _T, is_visible: bool) -> _T:
        """
        Returns a new NDBoundingBox object with the specified visibility.

        Args:
        - is_visible (bool): The visibility value to set.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated visibility value.
        """
        return attr.evolve(self, is_visible=is_visible)

    def with_color(self: _T, color: Optional[Tuple[float, float, float, float]]) -> _T:
        """
        Returns a new instance of NDBoundingBox with the specified color.

        Args:
        - color (Optional[Tuple[float, float, float, float]]): The color to set for the bounding box.
            The color should be specified as a tuple of four floats representing RGBA values.

        Returns:
        - NDBoundingBox: A new instance of NDBoundingBox with the specified color.
        """
        return attr.evolve(self, color=color)

    def with_bounds(
        self: _T, axis: str, new_topleft: Optional[int], new_size: Optional[int]
    ) -> _T:
        """
        Returns a new NDBoundingBox object with updated bounds along the specified axis.

        Args:
        - axis (str): The name of the axis to update.
        - new_topleft (Optional[int]): The new value for the top-left coordinate along the specified axis.
        - new_size (Optional[int]): The new size along the specified axis.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with updated bounds.

        Raises:
        - ValueError: If the given axis name does not exist.

        """
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
        """
        Returns the bounds of the given axis.

        Args:
        - axis (str): The name of the axis to get the bounds for.

        Returns:
        - Tuple[int, int]: A tuple containing the top-left and bottom-right coordinates along the specified axis.
        """
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
        """
        Create an instance of NDBoundingBox from a dictionary representation.

        Args:
        - bbox (Dict): The dictionary representation of the bounding box.

        Returns:
        - NDBoundingBox: An instance of NDBoundingBox.

        Raises:
        - AssertionError: If additionalAxes are present but axisOrder is not provided.
        """

        topleft: Tuple[int, ...] = bbox["topLeft"]
        size: Tuple[int, ...] = (bbox["width"], bbox["height"], bbox["depth"])
        axes: Tuple[str, ...] = ("x", "y", "z")
        index: Tuple[int, ...] = (1, 2, 3)

        if "axisOrder" in bbox:
            axes = tuple(bbox["axisOrder"].keys())
            index = tuple(bbox["axisOrder"][axis] for axis in axes)

            if "additionalAxes" in bbox:
                assert (
                    "axisOrder" in bbox
                ), "If there are additionalAxes an axisOrder needs to be provided."
                for axis in bbox["additionalAxes"]:
                    topleft += (axis["bounds"][0],)
                    size += (axis["bounds"][1] - axis["bounds"][0],)
                    axes += (axis["name"],)
                    index += (axis["index"],)

        return cls(
            topleft=VecInt(topleft, axes=axes),
            size=VecInt(size, axes=axes),
            axes=axes,
            index=VecInt(index, axes=axes),
        )

    def to_wkw_dict(self) -> dict:
        """
        Converts the bounding box object to a json dictionary.

        Returns:
        - dict: A json dictionary representing the bounding box.
        """
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
        """
        Returns a dictionary representation of the bounding box.

        Returns:
        - dict: A dictionary representation of the bounding box.
        """
        return {
            "topleft": self.topleft.to_list(),
            "size": self.size.to_list(),
            "axes": self.axes,
        }

    def to_checkpoint_name(self) -> str:
        """
        Returns a string representation of the bounding box that can be used as a checkpoint name.

        Returns:
        - str: A string representation of the bounding box.
        """
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
        """
        Returns the size of the bounding box along the specified axis.

        Args:
        - axis_name (str): The name of the axis to get the size for.

        Returns:
        - int: The size of the bounding box along the specified axis.
        """
        try:
            index = self.axes.index(axis_name)
            return self.size[index]
        except ValueError as err:
            raise ValueError(
                f"Axis {axis_name} doesn't exist in NDBoundingBox."
            ) from err

    def _get_attr_xyz(self, attr_name: str) -> Vec3Int:
        axes = ("x", "y", "z")
        attr_3d = []

        for axis in axes:
            index = self.axes.index(axis)
            attr_3d.append(getattr(self, attr_name)[index])

        return Vec3Int(attr_3d)

    def _get_attr_with_replaced_xyz(self, attr_name: str, xyz: Vec3IntLike) -> VecInt:
        value = Vec3Int(xyz)
        axes = ("x", "y", "z")
        modified_attr = getattr(self, attr_name).to_list()

        for i, axis in enumerate(axes):
            index = self.axes.index(axis)
            modified_attr[index] = value[i]

        return VecInt(modified_attr, axes=self.axes)

    @property
    def topleft_xyz(self) -> Vec3Int:
        """The topleft corner of the bounding box regarding only x, y and z axis."""

        return self._get_attr_xyz("topleft")

    @property
    def size_xyz(self) -> Vec3Int:
        """The size of the bounding box regarding only x, y and z axis."""

        return self._get_attr_xyz("size")

    @property
    def bottomright_xyz(self) -> Vec3Int:
        """The bottomright corner of the bounding box regarding only x, y and z axis."""

        return self._get_attr_xyz("bottomright")

    @property
    def index_xyz(self) -> Vec3Int:
        """The index of x, y and z axis within the bounding box."""

        return self._get_attr_xyz("index")

    def with_topleft_xyz(self: _T, new_xyz: Vec3IntLike) -> _T:
        """
        Returns a new NDBoundingBox object with changed x, y and z coordinates of the topleft corner.

        Args:
        - new_xyz (Vec3IntLike): The new x, y and z coordinates for the topleft corner.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated x, y and z coordinates of the topleft corner.
        """
        new_topleft = self._get_attr_with_replaced_xyz("topleft", new_xyz)

        return self.with_topleft(new_topleft)

    def with_size_xyz(self: _T, new_xyz: Vec3IntLike) -> _T:
        """
        Returns a new NDBoundingBox object with changed x, y and z size.

        Args:
        - new_xyz (Vec3IntLike): The new x, y and z size for the bounding box.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated x, y and z size.
        """
        new_size = self._get_attr_with_replaced_xyz("size", new_xyz)

        return self.with_size(new_size)

    def with_bottomright_xyz(self: _T, new_xyz: Vec3IntLike) -> _T:
        """
        Returns a new NDBoundingBox object with changed x, y and z coordinates of the bottomright corner.

        Args:
        - new_xyz (Vec3IntLike): The new x, y and z coordinates for the bottomright corner.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated x, y and z coordinates of the bottomright corner.
        """
        new_bottomright = self._get_attr_with_replaced_xyz("bottomright", new_xyz)

        return self.with_bottomright(new_bottomright)

    def with_index_xyz(self: _T, new_xyz: Vec3IntLike) -> _T:
        """
        Returns a new NDBoundingBox object with changed x, y and z index.

        Args:
        - new_xyz (Vec3IntLike): The new x, y and z index for the bounding box.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the updated x, y and z index.
        """
        new_index = self._get_attr_with_replaced_xyz("index", new_xyz)

        return self.with_index(new_index)

    def _check_compatibility(self, other: "NDBoundingBox") -> None:
        """Checks if two bounding boxes are comparable. To be comparable they need the same number of axes, with same names and same order."""

        if self.axes != other.axes:
            raise ValueError(
                f"Operation with two bboxes is only possible if they have the same axes and axes order. {self.axes} != {other.axes}"
            )

    def padded_with_margins(
        self, margins_left: VecIntLike, margins_right: Optional[VecIntLike] = None
    ) -> "NDBoundingBox":
        raise NotImplementedError()

    def intersected_with(self: _T, other: _T, dont_assert: bool = False) -> _T:
        """
        Returns the intersection of two bounding boxes.

        If dont_assert is set to False, this method may return empty bounding boxes (size == (0, 0, 0))

        Args:
        - other (NDBoundingBox): The other bounding box to intersect with.
        - dont_assert (bool): If True, the method may return empty bounding boxes.

        Returns:
        - NDBoundingBox: The intersection of the two bounding boxes.
        """

        self._check_compatibility(other)
        topleft = self.topleft.pairmax(other.topleft)
        bottomright = self.bottomright.pairmin(other.bottomright)
        size = (bottomright - topleft).pairmax(VecInt.zeros(self.axes))

        intersection = attr.evolve(self, topleft=topleft, size=size)

        if not dont_assert:
            assert (
                not intersection.is_empty()
            ), f"No intersection between bounding boxes {self} and {other}."

        return intersection

    def extended_by(self: _T, other: _T) -> _T:
        """
        Returns the smallest bounding box that contains both bounding boxes.

        Args:
        - other (NDBoundingBox): The other bounding box to extend with.

        Returns:
        - NDBoundingBox: The smallest bounding box that contains both bounding boxes.
        """
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
        """
        Boolean check whether the boundung box is empty.

        Returns:
        - bool: True if the bounding box is empty, False otherwise.
        """
        return not self.size.is_positive(strictly_positive=True)

    def in_mag(self: _T, mag: Mag) -> _T:
        """
        Returns the bounding box in the given mag.

        Args:
        - mag (Mag): The magnification to convert the bounding box to.

        Returns:
        - NDBoundingBox: The bounding box in the given magnification.
        """
        mag_vec = mag.to_vec3_int()

        assert (
            self.topleft_xyz % mag_vec == Vec3Int.zeros()
        ), f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        assert (
            self.bottomright_xyz % mag_vec == Vec3Int.zeros()
        ), f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."

        return self.with_topleft_xyz(self.topleft_xyz // mag_vec).with_size_xyz(
            self.size_xyz // mag_vec
        )

    def from_mag_to_mag1(self: _T, from_mag: Mag) -> _T:
        """
        Returns the bounging box in the finest magnification (Mag(1)).

        Args:
        - from_mag (Mag): The current magnification of the bounding box.

        Returns:
        - NDBoundingBox: The bounding box in the given magnification.
        """
        mag_vec = from_mag.to_vec3_int()

        return self.with_topleft_xyz(self.topleft_xyz * mag_vec).with_size_xyz(
            self.size_xyz * mag_vec
        )

    def _align_with_mag_slow(self: _T, mag: Mag, ceil: bool = False) -> _T:
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        :argument ceil: If true, the bounding box is enlarged when necessary. If false, it's shrinked when necessary.
        """
        np_mag = mag.to_np()

        align = (  # noqa E731
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

    def align_with_mag(self: _T, mag: Union[Mag, Vec3Int], ceil: bool = False) -> _T:
        """
        Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        Args:
        - mag (Union[Mag, Vec3Int]): The magnification to align the bounding box to.
        - ceil (bool): If True, the bounding box is enlarged when necessary. If False, it's shrinked when necessary.

        Returns:
        - NDBoundingBox: The aligned bounding box.
        """
        # This does the same as _align_with_mag_slow, which is more readable.
        # Same behavior is asserted in test_align_with_mag_against_numpy_implementation
        mag_vec = mag.to_vec3_int() if isinstance(mag, Mag) else mag
        topleft = self.topleft_xyz
        bottomright = self.bottomright_xyz
        roundup = topleft if ceil else bottomright
        rounddown = bottomright if ceil else topleft
        margin_to_roundup = roundup % mag_vec
        aligned_roundup = roundup - margin_to_roundup
        margin_to_rounddown = (mag_vec - (rounddown % mag_vec)) % mag_vec
        aligned_rounddown = rounddown + margin_to_rounddown
        if ceil:
            return self.with_topleft_xyz(aligned_roundup).with_size_xyz(
                aligned_rounddown - aligned_roundup
            )
        else:
            return self.with_topleft_xyz(aligned_rounddown).with_size_xyz(
                aligned_roundup - aligned_rounddown
            )

    def contains(self, coord: VecIntLike) -> bool:
        """
        Check whether a point is inside of the bounding box.
        Note that the point may have float coordinates in the ndarray case

        Args:
        - coord (VecIntLike): The coordinates to check.

        Returns:
        - bool: True if the point is inside of the bounding box, False otherwise.
        """

        if isinstance(coord, np.ndarray):
            assert (
                coord.shape == (len(self.size),)
            ), f"Numpy array BoundingBox.contains must have shape ({len(self.size)},), got {coord.shape}."
            return cast(
                bool,
                np.all(coord >= self.topleft) and np.all(coord < self.bottomright),
            )
        else:
            # In earlier versions, we simply converted to ndarray to have
            # a unified calculation here, but this turned out to be a performance bottleneck.
            # Therefore, the contains-check is performed on the tuple here.
            coord = VecInt(coord, axes=self.axes)
            return all(
                self.topleft[i] <= coord[i] < self.bottomright[i]
                for i in range(len(self.axes))
            )

    def contains_bbox(self: _T, inner_bbox: _T) -> bool:
        """
        Check whether a bounding box is completely inside of the bounding box.

        Args:
        - inner_bbox (NDBoundingBox): The bounding box to check.

        Returns:
        - bool: True if the bounding box is completely inside of the bounding box, False otherwise.
        """
        self._check_compatibility(inner_bbox)
        return inner_bbox.intersected_with(self, dont_assert=True) == inner_bbox

    def chunk(
        self: _T,
        chunk_shape: VecIntLike,
        chunk_border_alignments: Optional[VecIntLike] = None,
    ) -> Generator[_T, None, None]:
        """
        Decompose the bounding box into smaller chunks of size `chunk_shape`.

        Chunks at the border of the bounding box might be smaller than chunk_shape.
        If `chunk_border_alignment` is set, all border coordinates
        *between two chunks* will be divisible by that value.

        Args:
        - chunk_shape (VecIntLike): The size of the chunks to generate.
        - chunk_border_alignments (Optional[VecIntLike]): The alignment of the chunk borders.


        Yields:
        - Generator[NDBoundingBox]: A generator of the chunks.
        """

        start = self.topleft.to_np()
        try:
            # If a 3D chunk_shape is given it is assumed that iteration over xyz is
            # intended. Therefore NDBoundingBoxes are generated that have a shape of
            # x: chunk_shape.x, y: chunk_shape.y, z: chunk_shape.z and 1 for all other
            # axes.
            chunk_shape = Vec3Int(chunk_shape)

            chunk_shape = (
                self.with_size(VecInt.ones(self.axes))
                .with_size_xyz(chunk_shape)
                .size.to_np()
            )
        except AssertionError:
            chunk_shape = VecInt(chunk_shape, axes=self.axes).to_np()

        start_adjust = VecInt.zeros(self.axes).to_np()
        if chunk_border_alignments is not None:
            try:
                chunk_border_alignments = Vec3Int(chunk_border_alignments)

                chunk_border_alignments = (
                    self.with_size(VecInt.ones(self.axes))
                    .with_size_xyz(chunk_border_alignments)
                    .size.to_np()
                )
            except AssertionError:
                chunk_border_alignments = VecInt(
                    chunk_border_alignments, axes=self.axes
                ).to_np()

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
                self.__class__(
                    topleft=VecInt(coordinates, axes=self.axes),
                    size=VecInt(chunk_shape, axes=self.axes),
                    axes=self.axes,
                    index=self.index,
                )
            )

    def volume(self) -> int:
        """
        Returns the volume of the bounding box.
        """
        return self.size.prod()

    def slice_array(self, array: np.ndarray) -> np.ndarray:
        """
        Returns a slice of the given array that corresponds to the bounding box.
        """
        return array[self.to_slices()]

    def to_slices(self) -> Tuple[slice, ...]:
        """
        Returns a tuple of slices that corresponds to the bounding box.
        """
        return tuple(
            slice(topleft, topleft + size)
            for topleft, size in zip(self.topleft, self.size)
        )

    def offset(self: _T, vector: VecIntLike) -> _T:
        """
        Returns a new NDBoundingBox object with the specified offset.

        Args:
        - vector (VecIntLike): The offset to apply to the bounding box.

        Returns:
        - NDBoundingBox: A new NDBoundingBox object with the specified offset.
        """
        try:
            return self.with_topleft_xyz(self.topleft_xyz + Vec3Int(vector))
        except AssertionError:
            return self.with_topleft(self.topleft + VecInt(vector, axes=self.axes))

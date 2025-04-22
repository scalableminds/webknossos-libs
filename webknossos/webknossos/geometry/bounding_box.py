import json
import re
from collections.abc import Callable, Generator, Iterable
from typing import Union, cast

import attr
import numpy as np

from .mag import Mag
from .nd_bounding_box import NDBoundingBox
from .vec3_int import Vec3Int, Vec3IntLike

_DEFAULT_BBOX_NAME = "Unnamed Bounding Box"


@attr.frozen
class BoundingBox(NDBoundingBox):
    """An axis-aligned 3D bounding box with integer coordinates.

    This class represents a axis-aligned cuboid in 3D space. The top-left coordinate is
    inclusive and the bottom-right coordinate is exclusive, defining a volume in space.

    A small usage example:

    ```python
    from webknossos import BoundingBox

    bbox_1 = BoundingBox((0, 0, 0), (100, 100, 100))
    bbox_2 = BoundingBox((75, 75, 75), (100, 100, 100))

    assert bbox_1.intersected_with(bbox_2).size == (25, 25, 25)
    ```

    Attributes:
        topleft (Vec3Int): Top-left corner coordinates (inclusive)
        size (Vec3Int): Size of the bounding box in units of voxels for each dimension (width, height, depth)
        axes (tuple[str, str, str]): Names of the coordinate axes, defaults to ("x", "y", "z")
        index (Vec3Int): Index values for each dimension, defaults to (1, 2, 3)
        bottomright (Vec3Int): Bottom-right corner coordinates (exclusive), computed from topleft + size
        name (str | None): Optional name for the bounding box, defaults to "Unnamed Bounding Box"
        is_visible (bool): Whether the bounding box should be visible, defaults to True
        color (tuple[float, float, float, float] | None): Optional RGBA color values
    """

    topleft: Vec3Int = attr.field(converter=Vec3Int)
    size: Vec3Int = attr.field(converter=Vec3Int)
    axes: tuple[str, str, str] = attr.field(default=("x", "y", "z"))
    index: Vec3Int = attr.field(default=Vec3Int(1, 2, 3))
    bottomright: Vec3Int = attr.field(init=False)
    name: str | None = _DEFAULT_BBOX_NAME
    is_visible: bool = True
    color: tuple[float, float, float, float] | None = None

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

    def with_bounds_x(
        self, new_topleft_x: int | None = None, new_size_x: int | None = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.x optionally replaced and size.x optionally replaced."""

        return cast(BoundingBox, self.with_bounds("x", new_topleft_x, new_size_x))

    def with_bounds_y(
        self, new_topleft_y: int | None = None, new_size_y: int | None = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.y optionally replaced and size.y optionally replaced."""

        return cast(BoundingBox, self.with_bounds("y", new_topleft_y, new_size_y))

    def with_bounds_z(
        self, new_topleft_z: int | None = None, new_size_z: int | None = None
    ) -> "BoundingBox":
        """Returns a copy of the bounding box with topleft.z optionally replaced and size.z optionally replaced."""

        return cast(BoundingBox, self.with_bounds("z", new_topleft_z, new_size_z))

    @classmethod
    def from_wkw_dict(cls, bbox: dict) -> "BoundingBox":
        """Creates a BoundingBox from a wkw-format dictionary.

        Args:
            bbox (Dict): Dictionary containing wkw-format bounding box data with
                keys 'topLeft', 'width', 'height', and 'depth'

        Returns:
            BoundingBox: A new bounding box with the specified dimensions
        """
        return cls(bbox["topLeft"], [bbox["width"], bbox["height"], bbox["depth"]])

    @classmethod
    def from_config_dict(cls, bbox: dict) -> "BoundingBox":
        """Creates a BoundingBox from a config-format dictionary.

        Args:
            bbox (Dict): Dictionary containing config-format bounding box data with
                keys 'topleft' and 'size'

        Returns:
            BoundingBox: A new bounding box with the specified dimensions
        """
        return cls(bbox["topleft"], bbox["size"])

    @classmethod
    def from_tuple6(cls, tuple6: tuple[int, int, int, int, int, int]) -> "BoundingBox":
        """Creates a BoundingBox from a 6-tuple of coordinates.

        Args:
            tuple6 (tuple[int, int, int, int, int, int]): A tuple containing
                (x, y, z) coordinates followed by (width, height, depth) dimensions

        Returns:
            BoundingBox: A new bounding box with the specified dimensions
        """
        return cls(tuple6[0:3], tuple6[3:6])

    @classmethod
    def from_tuple2(cls, tuple2: tuple[Vec3IntLike, Vec3IntLike]) -> "BoundingBox":
        """Creates a BoundingBox from a 2-tuple of coordinates.

        Args:
            tuple2 (tuple[Vec3IntLike, Vec3IntLike]): A tuple containing
                the topleft coordinates and size dimensions

        Returns:
            BoundingBox: A new bounding box with the specified dimensions
        """
        return cls(tuple2[0], tuple2[1])

    @classmethod
    def from_points(cls, points: Iterable[Vec3IntLike]) -> "BoundingBox":
        """Returns a bounding box which is guaranteed to completely enclose all points in the input.

        Args:
            points (Iterable[Vec3IntLike]): Set of points to be bounded. Each point must be convertible to Vec3Int.

        Returns:
            BoundingBox: A bounding box that is the minimum size needed to contain all input points.exactly containing all points."""

        all_points = np.array([Vec3Int(point).to_list() for point in points])
        topleft = all_points.min(axis=0)
        bottomright = all_points.max(axis=0)

        # bottomright is exclusive
        bottomright += 1

        return cls(topleft, bottomright - topleft)

    @classmethod
    def from_checkpoint_name(cls, checkpoint_name: str) -> "BoundingBox":
        """This function extracts a bounding box in the format `x_y_z_sx_sy_xz` which is contained in a string."""
        regex = r"(([0-9]+_){5}([0-9]+))"
        match = re.search(regex, checkpoint_name)
        assert match is not None, (
            f"Could not extract bounding box from {checkpoint_name}"
        )
        bbox_tuple = tuple(int(value) for value in match.group().split("_"))
        return cls.from_tuple6(cast(tuple[int, int, int, int, int, int], bbox_tuple))

    @classmethod
    def from_csv(cls, csv_bbox: str) -> "BoundingBox":
        bbox_tuple = tuple(int(x) for x in csv_bbox.split(","))
        return cls.from_tuple6(cast(tuple[int, int, int, int, int, int], bbox_tuple))

    @classmethod
    def from_ndbbox(cls, bbox: NDBoundingBox) -> "BoundingBox":
        return cls(bbox.topleft_xyz, bbox.size_xyz)

    @classmethod
    def from_auto(
        cls, obj: Union["BoundingBox", str, dict, list, tuple]
    ) -> "BoundingBox":
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, str):
            if ":" in obj:
                return cls.from_auto(json.loads(obj))
            else:
                return cls.from_csv(obj)
        elif isinstance(obj, dict):
            if "size" in obj:
                return cls.from_config_dict(obj)
            return cls.from_wkw_dict(obj)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 2:
                return cls.from_tuple2(obj)  # type: ignore
            elif len(obj) == 6:
                return cls.from_tuple6(obj)  # type: ignore

        raise Exception("Unknown bounding box format.")

    @classmethod
    def empty(
        cls,
    ) -> "BoundingBox":
        return cls(Vec3Int.zeros(), Vec3Int.zeros())

    def to_wkw_dict(self) -> dict:
        """Converts the bounding box to a wkw-format dictionary.

        Creates a dictionary with wkw-format fields containing the bounding box dimensions.

        Returns:
            dict: A dictionary with keys:
                - topLeft: list[int] of (x,y,z) coordinates
                - width: int width in x dimension
                - height: int height in y dimension
                - depth: int depth in z dimension
        """
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
        """Converts the bounding box to a config-format dictionary.

        Creates a dictionary with config-format fields containing the bounding box dimensions.

        Returns:
            dict: A dictionary with keys:
                - topleft: list[int] of (x,y,z) coordinates
                - size: list[int] of (width,height,depth) dimensions
        """
        return {"topleft": self.topleft.to_list(), "size": self.size.to_list()}

    def to_checkpoint_name(self) -> str:
        """Converts the bounding box dimensions to a checkpoint name string.

        Creates a string formatted as "x_y_z_width_height_depth" containing the
        bounding box coordinates and dimensions.

        Returns:
            str: A string in checkpoint name format containing the bounding box dimensions
        """
        x, y, z = self.topleft
        width, height, depth = self.size
        return f"{x}_{y}_{z}_{width}_{height}_{depth}"

    def to_tuple6(self) -> tuple[int, int, int, int, int, int]:
        """Converts the bounding box coordinates to a 6-tuple.

        Creates a tuple containing the bounding box coordinates and dimensions.

        Returns:
            tuple[int, int, int, int, int, int]: A tuple containing:
                - First three values: (x,y,z) coordinates of topleft
                - Last three values: (width,height,depth) dimensions
        """
        return tuple(self.topleft.to_list() + self.size.to_list())  # type: ignore

    def to_csv(self) -> str:
        """Converts the bounding box coordinates to a comma-separated string.

        Creates a string containing the bounding box coordinates and dimensions
        in comma-separated format.

        Returns:
            str: A comma-separated string containing:
                - First three values: (x,y,z) coordinates of topleft
                - Last three values: (width,height,depth) dimensions
        """
        return ",".join(map(str, self.to_tuple6()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NDBoundingBox):
            self._check_compatibility(other)
            return self.topleft == other.topleft and self.size == other.size

        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"BoundingBox(topleft={self.topleft.to_tuple()}, size={self.size.to_tuple()})"

    def is_empty(self) -> bool:
        """Checks if the bounding box has zero or negative size.

        Tests if any dimension of the bounding box has zero or negative size.

        Returns:
            bool: True if any dimension has zero or negative size, False otherwise.
        """
        return not self.size.is_positive(strictly_positive=True)

    def in_mag(self, mag: Mag) -> "BoundingBox":
        """Returns a new bounding box with coordinates scaled by the given magnification factor.

        The method asserts that both topleft and bottomright coordinates are already properly
        aligned with the magnification factor. Use align_with_mag() first if needed.

        Args:
            mag (Mag): The magnification factor to scale coordinates by

        Returns:
            BoundingBox: A new bounding box with coordinates divided by the magnification factor

        Raises:
            AssertionError: If topleft or bottomright coordinates are not aligned with mag
        """
        mag_vec = mag.to_vec3_int()

        assert self.topleft % mag_vec == Vec3Int.zeros(), (
            f"topleft {self.topleft} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        )
        assert self.bottomright % mag_vec == Vec3Int.zeros(), (
            f"bottomright {self.bottomright} is not aligned with the mag {mag}. Use BoundingBox.align_with_mag()."
        )

        return attr.evolve(
            self,
            topleft=(self.topleft // mag_vec),
            size=(self.size // mag_vec),
        )

    def _align_with_mag_slow(self, mag: Mag, ceil: bool = False) -> "BoundingBox":
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        Args:
            mag (Mag): The mag to align with.
            ceil (bool): If true, the bounding box is enlarged when necessary. If false, it's shrunk when necessary.

        Returns:
            BoundingBox: The aligned bounding box.
        """
        np_mag = mag.to_np()

        def align(point: Vec3Int, round_fn: Callable) -> Vec3Int:
            return round_fn(point.to_np() / np_mag).astype(int) * np_mag

        if ceil:
            topleft = align(self.topleft, np.floor)
            bottomright = align(self.bottomright, np.ceil)
        else:
            topleft = align(self.topleft, np.ceil)
            bottomright = align(self.bottomright, np.floor)
        return attr.evolve(self, topleft=topleft, size=bottomright - topleft)

    def align_with_mag(self, mag: Mag | Vec3Int, ceil: bool = False) -> "BoundingBox":
        """Rounds the bounding box, so that both topleft and bottomright are divisible by mag.

        Args:
            mag (Mag): The mag to align with.
            ceil (bool): If true, the bounding box is enlarged when necessary. If false, it's shrunk when necessary.

        Returns:
            BoundingBox: The aligned bounding box.
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

    def contains(self, coord: Vec3IntLike | np.ndarray) -> bool:
        """Check whether a point is inside of the bounding box.
        Note that the point may have float coordinates in the ndarray case"""

        if isinstance(coord, np.ndarray):
            assert coord.shape == (3,), (
                f"Numpy array BoundingBox.contains must have shape (3,), got {coord.shape}."
            )
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

    def chunk(
        self,
        chunk_shape: Vec3IntLike,
        chunk_border_alignments: Vec3IntLike | None = None,
    ) -> Generator["BoundingBox", None, None]:
        """Decompose the bounding box into smaller chunks of size `chunk_shape`.

        Args:
            chunk_shape (Vec3IntLike): Size of chunks to decompose into. Each chunk
                will be at most this size.
            chunk_border_alignments (Vec3IntLike | None): If provided, all border
                coordinates between chunks will be divisible by these values.

        Yields:
            BoundingBox: Smaller chunks of the original bounding box. Border chunks
                may be smaller than chunk_shape.

        Raises:
            AssertionError: If chunk_border_alignments is provided and chunk_shape is
                not divisible by it.

        Note:
            - Border chunks may be smaller than chunk_shape
            - If chunk_border_alignments is provided, all border coordinates between
              chunks will be aligned to those values
        """

        start = self.topleft.to_np()
        chunk_shape = Vec3Int(chunk_shape).to_np()

        start_adjust = np.array([0, 0, 0])
        if chunk_border_alignments is not None:
            chunk_border_alignments_array = Vec3Int(chunk_border_alignments).to_np()
            assert np.all(chunk_shape % chunk_border_alignments_array == 0), (
                f"{chunk_shape} not divisible by {chunk_border_alignments_array}"
            )

            # Move the start to be aligned correctly. This doesn't actually change
            # the start of the first chunk, because we'll intersect with `self`,
            # but it'll lead to all chunk borders being aligned correctly.
            start_adjust = start % chunk_border_alignments_array

        for x in range(
            start[0] - start_adjust[0], start[0] + self.size[0], chunk_shape[0]
        ):
            for y in range(
                start[1] - start_adjust[1], start[1] + self.size[1], chunk_shape[1]
            ):
                for z in range(
                    start[2] - start_adjust[2], start[2] + self.size[2], chunk_shape[2]
                ):
                    yield cast(
                        BoundingBox,
                        BoundingBox([x, y, z], chunk_shape).intersected_with(self),
                    )

    def offset(self, vector: Vec3IntLike) -> "BoundingBox":
        """Creates an offset copy of this bounding box by adding a vector to the topleft coordinate.

        Generates a new bounding box with identical dimensions but translated by the given vector.

        Args:
            vector (Vec3IntLike): The vector to offset the bounding box by

        Returns:
            BoundingBox: A new bounding box offset by the given vector
        """
        return attr.evolve(self, topleft=self.topleft + Vec3Int(vector))

    def __hash__(self) -> int:
        return hash(self.to_tuple6())

"""Coordinate transformations of a dataset layer.

Coordinate transformations describe how a layer is placed into the shared coordinate
space of its dataset, e.g. to register several layers onto each other. They are pure
metadata: WEBKNOSSOS applies them while rendering, the voxel data on disk is left
untouched.

WEBKNOSSOS supports two kinds of transformations, `AffineCoordinateTransformation` and
`ThinPlateSplineCoordinateTransformation`. A layer holds a list of them, which are
applied in order.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import attr
import numpy as np

from ..geometry import X_AXIS, Y_AXIS, Z_AXIS, Vec3Float, Vec3FloatLike

Axis = Literal["x", "y", "z"]

_AXIS_TO_INDEX: dict[str, int] = {X_AXIS: 0, Y_AXIS: 1, Z_AXIS: 2}


def _axis_index(axis: Axis) -> int:
    try:
        return _AXIS_TO_INDEX[axis]
    except KeyError:
        raise ValueError(f"The axis must be `x`, `y` or `z`, got {axis!r}.") from None


def _read_only_array(value: Any) -> np.ndarray:
    # `np.array` always copies, so that neither the caller's array is turned read-only
    # nor the stored one can be written through the caller's reference
    array = np.array(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _as_matrix(value: Any) -> np.ndarray:
    matrix = _read_only_array(value)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"The affine matrix must have shape (4, 4), got {matrix.shape}."
        )
    return matrix


def _as_points(value: Any) -> np.ndarray:
    points = _read_only_array(value)
    if points.size == 0:
        raise ValueError(
            "A thin plate spline transformation needs at least one correspondence, "
            + "got an empty list of landmarks."
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"The correspondences must have shape (N, 3), got {points.shape}."
        )
    return points


class CoordinateTransformation(ABC):
    """Base class of the coordinate transformations of a layer.

    Transformations are immutable values: their arrays are read-only and their
    attributes cannot be reassigned. To change a transformation, build a new one and
    assign it to `Layer.coordinate_transformations`. Copying one therefore hands back
    the very same object.
    """

    def __copy__(self) -> "CoordinateTransformation":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "CoordinateTransformation":
        return self

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Serializes this transformation to its datasource-properties.json representation."""
        ...


@attr.define
class AffineCoordinateTransformation(CoordinateTransformation):
    """An affine transformation, given as a 4x4 homogeneous matrix.

    The matrix uses the usual mathematical convention, i.e. `matrix[row][column]` with
    the translation in the last column. Applied to a point `p`, it yields
    `matrix[:3, :3] @ p + matrix[:3, 3]`. For example, the matrix

    ```
    [[1, 0, 0, 10],
     [0, 1, 0, 20],
     [0, 0, 1, 30],
     [0, 0, 0,  1]]
    ```

    translates the layer by `(10, 20, 30)`.

    Instead of writing the matrix by hand, it can be built up from the `identity`,
    `from_translation`, `from_scale` and `from_rotation` constructors together with the
    `translate`, `scale`, `rotate`, `flip` and `chain` methods. Those methods return a
    new transformation and never modify the one they are called on. Each of them is
    applied after the transformation it is called on, so

    ```
    AffineCoordinateTransformation.identity().rotate("z", 90).translate((5, 0, 0))
    ```

    rotates the layer first and translates the result afterwards.
    """

    matrix: np.ndarray = attr.field(
        converter=_as_matrix,
        eq=attr.cmp_using(eq=np.array_equal),
        on_setattr=attr.setters.frozen,
    )
    """The 4x4 homogeneous transformation matrix as a read-only float64 numpy array."""

    @classmethod
    def identity(cls) -> "AffineCoordinateTransformation":
        """Creates a transformation that leaves the layer where it is."""
        return cls(matrix=np.eye(4))

    @classmethod
    def from_translation(
        cls, translation: Vec3FloatLike
    ) -> "AffineCoordinateTransformation":
        """Creates a transformation that translates the layer by `translation`."""
        matrix = np.eye(4)
        matrix[:3, 3] = Vec3Float(translation)
        return cls(matrix=matrix)

    @classmethod
    def from_scale(cls, scale: Vec3FloatLike) -> "AffineCoordinateTransformation":
        """Creates a transformation that scales the layer by `scale` around the origin."""
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag(Vec3Float(scale))
        return cls(matrix=matrix)

    @classmethod
    def from_rotation(
        cls, axis: Axis, angle: float
    ) -> "AffineCoordinateTransformation":
        """Creates a transformation that rotates the layer around the origin.

        Args:
            axis: The axis to rotate around, one of `"x"`, `"y"` or `"z"`.
            angle: The rotation angle in degrees, counter-clockwise when looking
                from the positive end of `axis` towards the origin.
        """
        index = _axis_index(axis)
        radians = np.deg2rad(angle)
        cosine = np.cos(radians)
        sine = np.sin(radians)
        # The two axes spanning the plane that is rotated
        first = (index + 1) % 3
        second = (index + 2) % 3
        matrix = np.eye(4)
        matrix[first, first] = cosine
        matrix[first, second] = -sine
        matrix[second, first] = sine
        matrix[second, second] = cosine
        return cls(matrix=matrix)

    def chain(
        self, other: "AffineCoordinateTransformation"
    ) -> "AffineCoordinateTransformation":
        """Returns a new transformation that applies this one first and `other` afterwards."""
        return type(self)(matrix=other.matrix @ self.matrix)

    def translate(self, translation: Vec3FloatLike) -> "AffineCoordinateTransformation":
        """Returns a new transformation that additionally translates by `translation`."""
        return self.chain(self.from_translation(translation))

    def scale(self, scale: Vec3FloatLike) -> "AffineCoordinateTransformation":
        """Returns a new transformation that additionally scales by `scale` around the origin."""
        return self.chain(self.from_scale(scale))

    def rotate(self, axis: Axis, angle: float) -> "AffineCoordinateTransformation":
        """Returns a new transformation that additionally rotates by `angle` degrees around `axis`.

        See `from_rotation` for the orientation of the rotation.
        """
        return self.chain(self.from_rotation(axis, angle))

    def flip(self, axis: Axis) -> "AffineCoordinateTransformation":
        """Returns a new transformation that additionally mirrors along `axis`.

        The layer is mirrored at the origin, i.e. the coordinates along `axis` change
        their sign.
        """
        scale = [1.0, 1.0, 1.0]
        scale[_axis_index(axis)] = -1.0
        return self.chain(self.from_scale(scale))

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "affine", "matrix": self.matrix.tolist()}


@attr.define
class ThinPlateSplineCoordinateTransformation(CoordinateTransformation):
    """A non-linear transformation defined by pairs of corresponding landmarks.

    `source[i]` is mapped onto `target[i]`; in between, the layer is warped smoothly by
    a thin plate spline. Both arrays must have the same length.
    """

    source: np.ndarray = attr.field(
        converter=_as_points,
        eq=attr.cmp_using(eq=np.array_equal),
        on_setattr=attr.setters.frozen,
    )
    """The landmarks in the layer's own coordinate space, as a read-only (N, 3) float64
    array."""

    target: np.ndarray = attr.field(
        converter=_as_points,
        eq=attr.cmp_using(eq=np.array_equal),
        on_setattr=attr.setters.frozen,
    )
    """The landmarks in the target coordinate space, as a read-only (N, 3) float64
    array."""

    def __attrs_post_init__(self) -> None:
        if len(self.source) != len(self.target):
            raise ValueError(
                "The source and target correspondences must have the same length, "
                + f"got {len(self.source)} and {len(self.target)}."
            )

    @classmethod
    def from_pairs(
        cls, pairs: Sequence[Sequence[Vec3FloatLike]]
    ) -> "ThinPlateSplineCoordinateTransformation":
        """Creates a transformation from `(source, target)` correspondence pairs.

        Args:
            pairs: The correspondences, each a pair of a source and a target point.
                The points may be `Vec3Int`, tuples, numpy arrays or any other iterable
                of three numbers.
        """
        source = []
        target = []
        for pair in pairs:
            pair = list(pair)
            if len(pair) != 2:
                raise ValueError(
                    "Each correspondence must be a pair of a source and a target "
                    + f"point, got {len(pair)} points."
                )
            source.append(Vec3Float(pair[0]))
            target.append(Vec3Float(pair[1]))
        return cls(source=source, target=target)

    @property
    def pairs(self) -> tuple[tuple[Vec3Float, Vec3Float], ...]:
        """The correspondences as `(source, target)` pairs."""
        return tuple(
            (Vec3Float(source), Vec3Float(target))
            for source, target in zip(self.source, self.target, strict=True)
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "type": "thin_plate_spline",
            "correspondences": {
                "source": self.source.tolist(),
                "target": self.target.tolist(),
            },
        }


def _coordinate_transformation_from_dict(d: dict[str, Any]) -> CoordinateTransformation:
    transformation_type = d.get("type")
    if transformation_type == "affine":
        return AffineCoordinateTransformation(matrix=d["matrix"])
    elif transformation_type == "thin_plate_spline":
        correspondences = d["correspondences"]
        return ThinPlateSplineCoordinateTransformation(
            source=correspondences["source"], target=correspondences["target"]
        )
    else:
        raise ValueError(
            "Failed to read a coordinate transformation of a layer: the type has to be "
            + f"`affine` or `thin_plate_spline`, got {transformation_type!r}."
        )

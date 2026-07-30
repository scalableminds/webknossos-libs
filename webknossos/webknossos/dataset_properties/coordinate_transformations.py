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
from typing import Any

import attr
import numpy as np


def _as_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"The affine matrix must have shape (4, 4), got {matrix.shape}."
        )
    return matrix


def _as_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"The correspondences must have shape (N, 3), got {points.shape}."
        )
    return points


class CoordinateTransformation(ABC):
    """Base class of the coordinate transformations of a layer.

    See `AffineCoordinateTransformation` and `ThinPlateSplineCoordinateTransformation`
    for the concrete transformations.
    """

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
    """

    matrix: np.ndarray = attr.field(
        converter=_as_matrix, eq=attr.cmp_using(eq=np.array_equal)
    )
    """The 4x4 homogeneous transformation matrix as a float64 numpy array."""

    @classmethod
    def identity(cls) -> "AffineCoordinateTransformation":
        """Creates a transformation that leaves the layer where it is."""
        return cls(matrix=np.eye(4))

    @classmethod
    def from_translation(
        cls, translation: Sequence[float]
    ) -> "AffineCoordinateTransformation":
        """Creates a transformation that translates the layer by `translation`."""
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return cls(matrix=matrix)

    @classmethod
    def from_scale(cls, scale: Sequence[float]) -> "AffineCoordinateTransformation":
        """Creates a transformation that scales the layer by `scale` around the origin."""
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag(np.asarray(scale, dtype=np.float64))
        return cls(matrix=matrix)

    def _to_dict(self) -> dict[str, Any]:
        return {"type": "affine", "matrix": self.matrix.tolist()}


@attr.define
class ThinPlateSplineCoordinateTransformation(CoordinateTransformation):
    """A non-linear transformation defined by pairs of corresponding landmarks.

    `source[i]` is mapped onto `target[i]`; in between, the layer is warped smoothly by
    a thin plate spline. Both arrays must have the same length.
    """

    source: np.ndarray = attr.field(
        converter=_as_points, eq=attr.cmp_using(eq=np.array_equal)
    )
    """The landmarks in the layer's own coordinate space, as an (N, 3) float64 array."""

    target: np.ndarray = attr.field(
        converter=_as_points, eq=attr.cmp_using(eq=np.array_equal)
    )
    """The landmarks in the target coordinate space, as an (N, 3) float64 array."""

    def __attrs_post_init__(self) -> None:
        if len(self.source) != len(self.target):
            raise ValueError(
                "The source and target correspondences must have the same length, "
                + f"got {len(self.source)} and {len(self.target)}."
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
        raise RuntimeError(
            "Failed to read a coordinate transformation of a layer: the type has to be "
            + f"`affine` or `thin_plate_spline`, got {transformation_type!r}."
        )

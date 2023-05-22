"""Utilities to work with wkcuber."""

from collections import namedtuple
from enum import Enum

from webknossos import BoundingBox, Mag


class DistributionStrategy(str, Enum):
    """Enum of available distribution strategies.

    TODO: As soon as supported by typer this enum should be
        replaced with typing.Literal in type hint.
    """

    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    MULTIPROCESSING = "multiprocessing"
    DEBUGS_SEQUENTIAL = "debug_sequential"


class DataFormat(str, Enum):
    """Enum of available data formats."""

    WKW = "wkw"
    ZARR = "zarr"


class SamplingMode(str, Enum):
    """Enum of available sampling modes."""

    ANISOTROPIC = "anisotropic"
    ISOTROPIC = "isotropic"
    CONSTANT_Z = "constant_z"


def parse_mag(mag_str: str) -> Mag:
    """Parses str input to Mag"""

    return Mag(mag_str)


VoxelSize = namedtuple("VoxelSize", ("x", "y", "z"))


def parse_voxel_size(voxel_size_str: str) -> VoxelSize:
    """Parses str input to tuple of three floats."""
    try:
        result = tuple(float(x) for x in voxel_size_str.split(","))
        if len(result) == 3:
            return VoxelSize(*result)
        raise ValueError(
            f"Expected three values formated like: 1.0,1.0,2.0 but got: {voxel_size_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to VoxelSize.\
Please format the voxel size like 1.0,1.0,2.0 ."
        ) from err


def parse_bbox(bbox_str: str) -> BoundingBox:
    """Parses str input to BoundingBox."""

    try:
        result = tuple(int(x) for x in bbox_str.split(","))
        if len(result) == 6:
            return BoundingBox.from_tuple6(
                (result[0], result[1], result[2], result[3], result[4], result[5])
            )
        raise ValueError(
            f"Expected six values formated like: 0,0,0,5,5,5 but got: {bbox_str}"
        )
    except Exception as err:
        raise ValueError(
            "The value could not be parsed to BoundingBox.\
Please format the bounding box like 0,0,0,5,5,5 ."
        ) from err

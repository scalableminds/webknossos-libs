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
        raise ValueError(f"Expected three values but got {len(result)}")
    except Exception as err:
        raise ValueError("The value could not be parsed to VoxelSize.") from err


def parse_bbox(bbox_str: str) -> BoundingBox:
    """Parses str input to BoundingBox."""

    try:
        result = tuple(int(x) for x in bbox_str.split(","))
        if len(result) == 6:
            return BoundingBox.from_tuple6(
                (result[0], result[1], result[2], result[3], result[4], result[5])
            )
        raise ValueError(f"Expected six values but got {len(result)}")
    except Exception as err:
        raise ValueError("The value could not be parsed to BoundingBox.") from err

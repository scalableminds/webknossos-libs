"""Utilities to work with wkcuber."""

from enum import Enum


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

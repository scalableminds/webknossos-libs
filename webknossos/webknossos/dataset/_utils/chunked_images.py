from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.nd_bounding_box import NDBoundingBox
from ..layer.view import MagView


class ChunkedImages(ABC):
    """
    Base class for volumetric, chunk-based input formats (Imaris .ims now;
    zarr, n5, neuroglancer_precomputed in the future).

    Unlike PimsImages, which reads slice-by-slice and can't always know its
    true x/y extent ahead of time (requiring a placeholder bounding box that
    gets corrected after conversion), a ChunkedImages implementation knows
    its exact expected_bbox from metadata alone, and reads/writes whole
    shard-sized 3D/4D blocks aligned to the output shard grid directly,
    without going through a slice-based writer.
    """

    dtype: DTypeLike
    num_channels: int

    def __init__(
        self,
        path: UPath,
        *,
        channel: int | None,  # noqa: ARG002 - documents the subclass constructor contract
        timepoint: int | None,  # noqa: ARG002
        swap_xy: bool,  # noqa: ARG002
        flip_x: bool,  # noqa: ARG002
        flip_y: bool,  # noqa: ARG002
        flip_z: bool,  # noqa: ARG002
        is_segmentation: bool,  # noqa: ARG002
    ) -> None:
        self.path = path

    @classmethod
    @abstractmethod
    def accepts(cls, path: UPath) -> bool:
        """Whether this class can read the given path."""

    @property
    @abstractmethod
    def channel(self) -> int | None:
        """The selected channel, or None if all channels are used."""

    @property
    @abstractmethod
    def expected_bbox(self) -> NDBoundingBox:
        """
        The exact bounding box of the data to convert, in the source's
        native Mag(1) space. Unlike PimsImages.expected_bbox, this never
        needs placeholder inflation, since chunk-based formats know their
        true extents from metadata alone.
        """

    @abstractmethod
    def get_possible_layers(self) -> dict[str, list[int]] | None:
        """Same semantics as PimsImages.get_possible_layers(): returns e.g.
        {"channel": [0, 1, 2]} when multiple channels could each become
        their own layer, or None if there's nothing to split."""

    @abstractmethod
    def read_chunk(
        self,
        bbox: NDBoundingBox,
        *,
        mag_view: MagView,
        dtype: DTypeLike | None,
    ) -> tuple[tuple[int, int], int]:
        """
        Read exactly bbox's worth of data and write it to mag_view directly.
        Returns ((x_size, y_size), max_value), matching the convention
        established by PimsImages.copy_to_view.
        """


_CHUNKED_IMAGE_CLASSES: list[type[ChunkedImages]] = []


def register_chunked_images(cls: type[ChunkedImages]) -> type[ChunkedImages]:
    _CHUNKED_IMAGE_CLASSES.append(cls)
    return cls


def try_open_chunked_images(
    images: object,
    *,
    channel: int | None,
    timepoint: int | None,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    is_segmentation: bool,
) -> ChunkedImages | None:
    """
    Returns a ChunkedImages instance if `images` is a single path that a
    registered chunk-based format can read, else None. Chunk-based formats
    are inherently single-file, so lists of paths and pims.FramesSequence
    instances always fall back to the generic PimsImages path.
    """
    if not isinstance(images, (str, UPath)):
        return None
    path = UPath(images)
    for cls in _CHUNKED_IMAGE_CLASSES:
        if cls.accepts(path):
            return cls(
                path,
                channel=channel,
                timepoint=timepoint,
                swap_xy=swap_xy,
                flip_x=flip_x,
                flip_y=flip_y,
                flip_z=flip_z,
                is_segmentation=is_segmentation,
            )
    return None


def _chunked_images_imports() -> None:
    try:
        from .ims_chunked_images import ImsChunkedImages  # noqa: F401
    except ImportError:
        pass  # the `ims` extra isn't installed; .ims files fall back to the generic pims path


_chunked_images_imports()

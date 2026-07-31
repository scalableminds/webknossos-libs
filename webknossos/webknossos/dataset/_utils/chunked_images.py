from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from os import environ

from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.nd_bounding_box import NDBoundingBox
from ..layer.view import MagView


class ChunkedImages(ABC):
    """
    Base class for volumetric, chunk-based input formats (Imaris .ims and
    MRC now; zarr, n5, neuroglancer_precomputed in the future).

    Unlike PimsImages, which reads slice-by-slice and can't always know its
    true x/y extent ahead of time (requiring a placeholder bounding box that
    gets corrected after conversion), a ChunkedImages implementation knows
    its exact expected_bbox from metadata alone, and reads/writes whole
    shard-sized 3D/4D blocks aligned to the output shard grid directly,
    without going through a slice-based writer.

    Formats handled by a registered ChunkedImages subclass are read
    exclusively through that subclass — never through PimsImages/pims,
    regardless of `use_bioformats`.
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
    def class_exts(cls) -> set[str]:
        """File extensions (without the dot, lowercase) this class can read."""

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


def get_valid_chunked_image_suffixes() -> set[str]:
    valid_suffixes: set[str] = set()
    for cls in _CHUNKED_IMAGE_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


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
    Returns a ChunkedImages instance if `images` is a single path whose
    suffix a registered chunk-based format handles, else None. Chunk-based
    formats are inherently single-file, so lists of paths and
    pims.FramesSequence instances always fall back to the generic PimsImages
    path.
    """
    if not isinstance(images, (str, UPath)):
        return None
    path = UPath(images)
    suffix = path.suffix.lstrip(".").lower()
    for cls in _CHUNKED_IMAGE_CLASSES:
        if suffix in cls.class_exts():
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


def _chunked_images_imports() -> str | None:
    import_exceptions = []

    try:
        from .ims_chunked_images import ImsChunkedImages  # noqa: F401 unused-import
    except ImportError as import_error:
        import_exceptions.append(f"ImsChunkedImages: {import_error.msg}")

    try:
        from .mrc_chunked_images import MrcChunkedImages  # noqa: F401 unused-import
    except ImportError as import_error:
        import_exceptions.append(f"MrcChunkedImages: {import_error.msg}")

    if import_exceptions:
        import_exception_string = "".join(
            f"\t- {import_exception}\n" for import_exception in import_exceptions
        )
        return import_exception_string
    return None


if (chunked_images_warnings := _chunked_images_imports()) is not None:
    if (
        environ.get("WEBKNOSSOS_SHOWED_CHUNKED_IMAGES_IMPORT_WARNING", "False")
        == "False"
    ):
        # If the environment variable is not set, we assume that the user has not seen the warning yet.
        # We set it to True to prevent showing the warning again.
        environ["WEBKNOSSOS_SHOWED_CHUNKED_IMAGES_IMPORT_WARNING"] = "True"
        warnings.warn(
            f"[WARNING] Not all chunk-based image readers could be imported:\n{chunked_images_warnings}Install the readers you need or use 'webknossos[all]' to install all readers.",
            category=UserWarning,
            source=None,
            stacklevel=2,
        )

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from os import environ

import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

from ...geometry.bounding_box import BoundingBox
from ...geometry.nd_bounding_box import NDBoundingBox
from ...geometry.vec_int import VecInt
from ..layer.view import MagView


class ChunkedImages(ABC):
    """
    Base class for volumetric, chunk-based input formats (e.g. ims, mrc, zarr, n5).

    Unlike SlicedImages, which reads slice-by-slice and can't always know its
    true x/y extent ahead of time (requiring a placeholder bounding box that
    gets corrected after conversion), a ChunkedImages implementation knows
    its exact expected_bbox from metadata alone, and reads/writes whole
    shard-sized 3D/4D blocks aligned to the output shard grid directly,
    without going through a slice-based writer.

    Formats handled by a registered ChunkedImages subclass are read
    exclusively through that subclass — never through SlicedImages.
    """

    dtype: DTypeLike
    num_channels: int

    # Source extents in the file's own axis order, which subclasses must set
    # from metadata in __init__. The base class derives expected_bbox and every
    # read range below from these, so they are always pre-swap_xy: _x is the
    # source's x, even when swap_xy makes it the output's y.
    _x: int
    _y: int
    _z: int
    # Timepoint handling. Formats without a time dimension leave these alone
    # and are read at timepoint 0; a subclass with several timepoints and none
    # pinned sets _include_t_axis, which puts a "t" axis on expected_bbox and
    # makes each chunk carry its own timepoint.
    _t: int = 1
    _include_t_axis: bool = False
    _fixed_timepoint: int | None = 0

    def __init__(
        self,
        path: UPath,
        *,
        channel: int | None,
        swap_xy: bool,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        is_segmentation: bool,
    ) -> None:
        self.path = path
        self._channel = channel
        self._swap_xy = swap_xy
        self._flip_x = flip_x
        self._flip_y = flip_y
        self._flip_z = flip_z
        self._is_segmentation = is_segmentation

    @classmethod
    @abstractmethod
    def class_exts(cls) -> set[str]:
        """File extensions (without the dot, lowercase) this class can read."""

    @property
    def channel(self) -> int | None:
        """The selected channel, or None if all channels are used."""
        return self._channel

    @abstractmethod
    def get_possible_layers(self) -> dict[str, list[int]] | None:
        """Same semantics as SlicedImages.get_possible_layers(): returns e.g.
        {"channel": [0, 1, 2]} when multiple channels could each become
        their own layer, or None if there's nothing to split."""

    @abstractmethod
    def _read_source_box(
        self,
        *,
        timepoint: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        """
        Read exactly the given box in *source* coordinates and return it as
        (c, z, y, x), with a leading channel axis even for single-channel
        formats. The slices are already resolved to the source's own axis
        order and mag, and already account for swap_xy and for flips (which
        arrive as mirrored ranges) — implementations do no axis bookkeeping of
        their own beyond selecting channels.
        """

    @property
    def expected_bbox(self) -> NDBoundingBox:
        """
        The exact bounding box of the data to convert, in the source's
        native Mag(1) space. Unlike SlicedImages.expected_bbox, this never
        needs placeholder inflation, since chunk-based formats know their
        true extents from metadata alone.

        Reports every axis the source actually has, including "c" when more
        than one channel is written (which is where NormalizedBoundingBox
        reads num_channels from) and "t" for unpinned multi-timepoint data.
        Always well-defined — it never has to reject an axis combination.
        """
        x_size, y_size = self._x, self._y
        if self._swap_xy:
            x_size, y_size = y_size, x_size

        if not self._include_t_axis and self.num_channels == 1:
            return BoundingBox((0, 0, 0), (x_size, y_size, self._z))

        # NDBoundingBox.chunk() keeps "c" whole rather than splitting it, so
        # read_chunk still receives the full channel extent per chunk.
        axes = ["x", "y", "z"]
        sizes = [x_size, y_size, self._z]
        if self.num_channels > 1:
            axes.insert(0, "c")
            sizes.insert(0, self.num_channels)
        if self._include_t_axis:
            axes.insert(0, "t")
            sizes.insert(0, self._t)
        return NDBoundingBox(
            VecInt.zeros(tuple(axes)),
            VecInt(sizes, axes=axes),
            axes,
            VecInt(list(range(len(axes))), axes=axes),
        )

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
        established by SlicedImages.copy_to_view.

        All axis bookkeeping — mag, swap_xy, flips, channel and timepoint
        placement — lives here, so subclasses only implement _read_source_box.
        """
        relative_bbox = bbox.offset(-mag_view.bounding_box.topleft)

        if "t" in relative_bbox.axes:
            timepoint, _ = relative_bbox.get_bounds("t")
        else:
            assert self._fixed_timepoint is not None
            timepoint = self._fixed_timepoint

        # bbox is in Mag(1) while the source is indexed in its own mag, so
        # every bound is scaled down before it addresses source data. For
        # Mag(1) this is the identity.
        mag_vec = mag_view.mag.to_vec3_int()

        # bbox's x/y axes describe the *output* extents. When swap_xy is set,
        # expected_bbox swaps which source axis feeds which output axis, so
        # bbox.x holds the source y-extent and bbox.y holds the source
        # x-extent — the read below must use the matching source bound.
        out_x_start, out_x_end = relative_bbox.get_bounds("x")
        out_y_start, out_y_end = relative_bbox.get_bounds("y")
        z_start, z_end = relative_bbox.get_bounds("z")
        z_start, z_end = z_start // mag_vec.z, z_end // mag_vec.z
        if self._swap_xy:
            source_y_start, source_y_end = (
                out_x_start // mag_vec.x,
                out_x_end // mag_vec.x,
            )
            source_x_start, source_x_end = (
                out_y_start // mag_vec.y,
                out_y_end // mag_vec.y,
            )
        else:
            source_x_start, source_x_end = (
                out_x_start // mag_vec.x,
                out_x_end // mag_vec.x,
            )
            source_y_start, source_y_end = (
                out_y_start // mag_vec.y,
                out_y_end // mag_vec.y,
            )

        # Every flip mirrors the *entire* source extent (matching
        # SlicedImages.copy_to_view, which reverses the full image sequence
        # before slicing per-batch), not just this chunk in isolation. Each one
        # reads a mirrored source range and is reversed back into output order
        # below. Reversing a chunk in place would only mirror within that
        # chunk, which is invisible while the image fits in a single shard but
        # wrong as soon as it spans several.
        # flip_x/-y follow the SlicedImages convention: flip_x mirrors the
        # source's y axis and flip_y mirrors its x axis.
        if self._flip_z:
            read_z = slice(self._z - z_end, self._z - z_start)
        else:
            read_z = slice(z_start, z_end)
        if self._flip_x:
            read_y = slice(self._y - source_y_end, self._y - source_y_start)
        else:
            read_y = slice(source_y_start, source_y_end)
        if self._flip_y:
            read_x = slice(self._x - source_x_end, self._x - source_x_start)
        else:
            read_x = slice(source_x_start, source_x_end)

        block = self._read_source_box(timepoint=timepoint, z=read_z, y=read_y, x=read_x)

        # Mirrored read ranges -> correct output order, in source axis order
        # (c, z, y, x).
        if self._flip_z:
            block = block[:, ::-1]
        if self._flip_x:
            block = block[:, :, ::-1]
        if self._flip_y:
            block = block[:, :, :, ::-1]

        # Transpose to mag_view.write() convention (c, x, y, z).
        # swap_xy=False (default): (c, z, y, x) → (c, x, y, z)
        # swap_xy=True:            (c, z, y, x) → (c, y, x, z)
        block = (
            block.transpose(0, 3, 2, 1)
            if not self._swap_xy
            else block.transpose(0, 2, 3, 1)
        )

        if dtype is not None:
            # order="F" matches SlicedImages.copy_to_view, which produces
            # Fortran-contiguous slices for the same downstream writers.
            block = block.astype(dtype, order="F")

        max_value = int(block.max())
        if self.num_channels == 1:
            block = block[0]  # (x, y, z) — single-channel layers have no c axis
        if "t" in relative_bbox.axes:
            # View.write() requires data.shape to have exactly as many
            # dimensions as the layer's bbox axes; add the (size-1) "t"
            # dimension this chunk corresponds to.
            block = block[np.newaxis]

        # allow_unaligned=True: real image extents rarely divide evenly into
        # full shards, so border chunks are smaller than shard_shape. Safe
        # here because each parallel job writes a disjoint,
        # shard-aligned-or-smaller region — no two jobs ever target
        # overlapping data.
        mag_view.write(block, absolute_bounding_box=bbox, allow_unaligned=True)

        # Returned as (x_size, y_size) to match the convention established by
        # SlicedImages.copy_to_view.
        return (out_x_end - out_x_start, out_y_end - out_y_start), max_value


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
    images: UPath | list[UPath],
    *,
    channel: int | None,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    is_segmentation: bool,
) -> ChunkedImages | None:
    """
    Returns a ChunkedImages instance if `images` is a single path whose
    suffix a registered chunk-based format handles, else None. Chunk-based
    formats are inherently single-file, so lists of paths always fall back to
    the generic SlicedImages path.
    """
    # Callers normalize paths to UPath before getting here (see
    # _normalize_images_argument), so a list of paths is by definition not a
    # single chunk-based file.
    # Remote UPaths deliberately do reach the reader, which raises its own
    # "must be a local file path" error via is_remote_path.
    if not isinstance(images, UPath):
        return None
    path = images
    suffix = path.suffix.lstrip(".").lower()
    for cls in _CHUNKED_IMAGE_CLASSES:
        if suffix in cls.class_exts():
            return cls(
                path,
                channel=channel,
                swap_xy=swap_xy,
                flip_x=flip_x,
                flip_y=flip_y,
                flip_z=flip_z,
                is_segmentation=is_segmentation,
            )
    return None


# The suffixes and extra each optional reader is responsible for. A reader
# whose dependency is missing never imports, so it never registers and cannot
# report its own class_exts() — these have to be declared out here for the
# "you are missing an optional dependency" hint to be possible at all.
# test_optional_reader_suffixes_match_class_exts keeps them in sync.
_OPTIONAL_CHUNKED_IMAGE_READERS: dict[str, tuple[str, frozenset[str]]] = {
    "ImsChunkedImages": ("ims", frozenset({"ims"})),
    "MrcChunkedImages": ("mrcfile", frozenset({"mrc", "rec", "st", "map", "ali"})),
}

# suffix -> extra, for readers that failed to import. Populated at import time
# below and consumed by get_unavailable_chunked_image_suffixes().
_UNAVAILABLE_CHUNKED_IMAGE_SUFFIXES: dict[str, str] = {}


def get_unavailable_chunked_image_suffixes() -> dict[str, str]:
    """
    Maps each suffix that a chunk-based reader *would* handle, but cannot
    because its optional dependency is missing, to the extra that provides it
    (e.g. {"ims": "ims"}). Empty when every reader imported successfully.

    Lets callers turn "no supported image data found" into a message that
    names the missing dependency, rather than silently omitting the format
    from the supported list.
    """
    return dict(_UNAVAILABLE_CHUNKED_IMAGE_SUFFIXES)


def describe_missing_extras(found: dict[str, str]) -> str:
    """
    Turns a suffix -> extra mapping, as returned for the files at hand by
    get_unavailable_chunked_image_suffixes(), into a sentence naming the
    extras to install. Meant to be appended to an error message.
    """
    extras = sorted(set(found.values()))
    return (
        f". Found {', '.join('.' + s for s in sorted(found))} files, which need an "
        + "optional dependency that is not installed — install it with "
        + f"`pip install {' '.join(f'webknossos[{extra}]' for extra in extras)}`"
    )


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

    registered = {cls.__name__ for cls in _CHUNKED_IMAGE_CLASSES}
    for name, (extra, suffixes) in _OPTIONAL_CHUNKED_IMAGE_READERS.items():
        if name not in registered:
            for suffix in suffixes:
                _UNAVAILABLE_CHUNKED_IMAGE_SUFFIXES[suffix] = extra

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

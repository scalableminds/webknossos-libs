from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from upath import UPath

from ...utils import WkImportError, is_remote_path
from ..errors import CorruptImageError, UnsupportedImageDataError
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_reader

try:
    from pylibCZIrw import czi as pyczi
except ImportError as e:
    raise WkImportError("pylibCZIrw", "czi") from e

# Each CZI pixel type fixes both the dtype of a voxel and how many colour
# components it carries. Gray* is a single component, Bgr* three, Bgra* four.
PIXEL_TYPES: dict[str, tuple[str, int]] = {
    "Gray8": ("uint8", 1),
    "Gray16": ("uint16", 1),
    "Gray32": ("int32", 1),
    "Gray64": ("int64", 1),
    "Gray32Float": ("float32", 1),
    "Gray64ComplexFloat": ("complex64", 1),
    "Bgr24": ("uint8", 3),
    "Bgr48": ("uint16", 3),
    "Bgr96Float": ("float32", 3),
    "Bgr192ComplexFloat": ("complex64", 3),
    "Bgra32": ("uint8", 4),
}

# Dimensions this reader can place. "C" is deliberately absent: a CZI "C" is a
# separate acquisition channel, selected with czi_channel, not a colour
# component of one image (those come from the pixel type instead).
_SUPPORTED_DIMS = frozenset({"X", "Y", "Z", "T", "C"})


@register_chunked_reader
class CziImageSource(ChunkedImageSource):
    """
    ChunkedImageSource for Zeiss .czi files. CZI stores its data as tiled
    subblocks and states its exact extents in metadata, so each chunk reads
    just the box it needs via pylibCZIrw's `roi` argument, one z-plane at a
    time, instead of decoding whole planes and writing them slice by slice.

    A CZI "C" dimension holds separate acquisition channels, which may even
    differ in pixel type. It is selected with `czi_channel` rather than being
    written as colour channels — the colour components of one image come from
    its pixel type (Gray/Bgr/Bgra).
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"czi"}

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        if is_remote_path(path):
            raise ValueError(
                f"Cannot open CZI file from {path}. The path must be a local file path."
            )
        # A format-specific option: only this reader knows what czi_channel
        # means, and it is the name get_possible_layers() reports it under.
        czi_channel = options.format_option("czi_channel")
        self.czi_channel = 0 if czi_channel is None else czi_channel

        with self._czi_file() as czi_file:
            bounding_box = czi_file.total_bounding_box
            self._available_czi_channels = sorted(czi_file.pixel_types.keys())

            # read() without an roi returns exactly this rectangle, so it is
            # the frame every roi below is relative to — not the X/Y entries of
            # total_bounding_box, which need not share its origin.
            rectangle = czi_file.total_bounding_rectangle
            self._x_offset, self._y_offset = rectangle.x, rectangle.y
            self._x, self._y = rectangle.w, rectangle.h

            self._z_offset, z_size = _dimension(bounding_box, "Z")
            self._z = z_size
            self._t_offset, self._t = _dimension(bounding_box, "T")

            # CZI can carry further dimensions (R, I, H, V, B — rotations,
            # illuminations, phases, views, blocks), which have no place in the
            # output bounding box. Length-1 ones are simply not addressed,
            # which pylibCZIrw resolves to their only value; anything longer
            # would silently drop data, so refuse it instead.
            #
            # Scenes are not among them: they are not a dimension at all, but
            # regions within total_bounding_rectangle, so a multi-scene file
            # converts to one image spanning all of them.
            for dim, (start, end) in bounding_box.items():
                if dim not in _SUPPORTED_DIMS and end - start > 1:
                    raise UnsupportedImageDataError(
                        f"Cannot convert {path}: it has a '{dim}' dimension of "
                        + f"size {end - start}, which cannot be represented. Only "
                        + "the X, Y, Z, T and C dimensions are supported.",
                        path=path,
                    )

            pixel_type = czi_file.get_channel_pixel_type(self.czi_channel)

        if pixel_type == "Invalid":
            raise ValueError(
                f"czi_channel {self.czi_channel} does not exist in {path}. "
                + f"Available channels: {self._available_czi_channels}."
            )
        if pixel_type not in PIXEL_TYPES:
            raise UnsupportedImageDataError(
                f"Got unsupported czi pixel-type {pixel_type} in {path}.",
                path=path,
            )
        dtype_name, raw_num_channels = PIXEL_TYPES[pixel_type]
        self.dtype = np.dtype(dtype_name)

        self.num_channels, self._channel, self._first_n_channels, possible_channels = (
            compute_channel_selection(raw_num_channels, options.channel)
        )
        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels
        if len(self._available_czi_channels) > 1:
            self._possible_layers["czi_channel"] = self._available_czi_channels

        self._include_t_axis = self._t > 1
        self._fixed_timepoint = None if self._include_t_axis else 0

    @contextmanager
    def _czi_file(self) -> Iterator[pyczi.CziReader]:
        try:
            czi_file = pyczi.open_czi(str(self.path))
        except Exception as e:
            raise CorruptImageError(
                f"Cannot open CZI file {self.path}. "
                + "The file is likely corrupted or not a valid CZI file.",
                path=self.path,
            ) from e
        with czi_file as opened:
            yield opened

    def get_possible_layers(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    def _read_source_box(
        self,
        *,
        timepoint: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        roi = (
            self._x_offset + x.start,
            self._y_offset + y.start,
            x.stop - x.start,
            y.stop - y.start,
        )
        planes = []
        with self._czi_file() as czi_file:
            for source_z in range(z.start, z.stop):
                plane = {
                    "C": self.czi_channel,
                    "Z": self._z_offset + source_z,
                    "T": self._t_offset + timepoint,
                }
                planes.append(czi_file.read(roi=roi, plane=plane))  # (y, x, c)

        block = np.moveaxis(np.stack(planes, axis=0), -1, 0)  # (c, z, y, x)

        if block.shape[0] in (3, 4):
            # bgr -> rgb / bgra -> rgba; both only swap the first and third
            # component, leaving green (and alpha) where they are.
            order = [2, 1, 0] + list(range(3, block.shape[0]))
            block = block[order]

        if self._channel is not None:
            block = block[self._channel : self._channel + 1]
        elif self._first_n_channels is not None:
            block = block[: self._first_n_channels]
        return block


def _dimension(bounding_box: dict[str, tuple[int, int]], name: str) -> tuple[int, int]:
    """The (offset, size) of a CZI dimension, or (0, 1) when the file has
    none — an absent dimension behaves exactly like one of length 1."""
    if name not in bounding_box:
        return 0, 1
    start, end = bounding_box[name]
    return start, end - start

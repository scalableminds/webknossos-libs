import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TypeVar, cast

import numpy as np
from natsort import natsorted
from numpy.typing import DTypeLike
from upath import UPath

from ...dataset_properties import DataFormat
from ...geometry.bounding_box import BoundingBox
from ...geometry.constants import C_AXIS, CXYZ_AXES, X_AXIS, Y_AXIS, Z_AXIS
from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
from ...geometry.normalized_bounding_box import NormalizedBoundingBox
from ...geometry.vec3_int import Vec3Int
from ..errors import (
    CorruptImageError,
    UnsupportedImageFormatError,
)
from ..layer.view import MagView
from .common_slice_readers import MultiImageSliceReader, StackedFileSliceReader
from .image_source import (
    ChunkResult,
    ImageSource,
    ReadOptions,
    compute_channel_selection,
    with_explicit_channel_axis,
)
from .image_source_registry import (
    describe_missing_extras,
    get_unavailable_extensions,
    get_valid_extensions,
    get_valid_slice_reader_extensions,
    open_slice_reader,
)
from .slice_reader import SliceReader, _SlicedView

# The x/y extent is only discovered while reading, so the layer starts out
# deliberately oversized and is cut back down once reading is complete.
SAFE_LARGE_XY: int = 10_000_000_000  # 10 billion


class SlicedImageSource(ImageSource):
    """
    ImageSource for formats read one 2D slice at a time (common raster image
    formats, TIFF, DM3/DM4) rather than in arbitrary boxes, via a
    `SliceReader`. Extents are not known upfront, so conversion starts with
    an oversized placeholder bbox and `final_bounding_box` corrects it from
    what was actually written.

    Composes a `SliceReader` rather than subclassing one, re-opening it for
    every chunk: sources are pickled across processes and must not carry open
    file handles between calls.
    """

    # Set once the axes are known; kept as bare annotations rather than
    # assignments so their presence can be checked with hasattr().
    _bundle_axes: list[str]
    _default_coords: dict[str, int]

    def __init__(
        self,
        image_paths: UPath | list[UPath],
        options: ReadOptions,
    ) -> None:
        """Opens the reader once to discover its dtype and axes, then works
        out how to present every slice and how channels are selected.

        Sets `dtype` and `channels_are_rgb` from the reader, and derives
        `_bundle_axes` (what one slice is — "y" and "x", preceded by "c" when
        there are several channels) and `_iter_axes` (what is stepped through
        per slice, "z" last so it varies fastest; empty for a single 2D
        image). `num_channels`, `_channel` and `_first_n_channels` come from
        `compute_channel_selection`, which also fills
        `_possible_layers["channel"]` when the channels could be split across
        layers.
        """
        # `images` below always refers to an opened reader, never to this
        # argument.
        self._original_images = image_paths
        self._options = options
        self._channel = options.channel  # replaced below by the resolved one
        self._iter_axes: list[str] = []
        self._possible_layers = {}

        with self._open_slice_reader() as images:
            self.dtype = images.dtype
            self.channels_are_rgb = images.channels_are_rgb
            self._default_coords = {}

            # A slice is a 2D image, channels first when there are several.
            raw_num_channels = images.sizes.get(C_AXIS, 1)
            if raw_num_channels > 1:
                bundle_axes = [C_AXIS, Y_AXIS, C_AXIS]
            else:
                if C_AXIS in images.axes:
                    # In neither list, so coordinate 0 is what gets returned.
                    self._default_coords[C_AXIS] = 0
                bundle_axes = [Y_AXIS, C_AXIS]

            # Every remaining axis is iterated over. "z" goes last, so it is
            # the fastest-varying one.
            self._iter_axes = sorted(
                set(images.axes).difference({*bundle_axes, C_AXIS, Z_AXIS})
            )
            if Z_AXIS in images.axes:
                self._iter_axes.append(Z_AXIS)
            self._bundle_axes = bundle_axes

        self.num_channels, self._channel, self._first_n_channels, possible_channels = (
            compute_channel_selection(raw_num_channels, self._channel)
        )
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

    def _resolve_original_image_paths(self) -> str | list[str]:
        original_images = self._original_images
        if isinstance(original_images, list):
            return [str(i) for i in original_images]
        if original_images.is_dir():
            valid_extensions = get_valid_slice_reader_extensions()
            files: list[str] = natsorted(
                str(i)
                for i in original_images.glob("**/*")
                if i.is_file() and i.suffix.lstrip(".") in valid_extensions
            )
            if len(files) == 1:
                return files[0]
            return files
        return str(original_images)

    def _error_path(self) -> UPath | None:
        """The single input path to blame when opening fails, or None when
        there is no such path (an empty list of images)."""
        if isinstance(self._original_images, UPath):
            return self._original_images
        if isinstance(self._original_images, list) and self._original_images:
            # A list is opened with a single reader, chosen from the first
            # image, so that one decides how the whole list is treated.
            return UPath(self._original_images[0])
        return None

    def _classify_open_failure(self) -> UnsupportedImageFormatError | None:
        """
        Decides whether a failure to open the images means "no reader supports
        this format" — in which case the returned exception should be raised —
        or something else, e.g. a corrupt file or an IO error of a format that
        *is* supported, which must keep surfacing as its original error.

        The decision is made on the extension, not the exception type, since
        an unsupported format and a corrupt file of a supported one raise the
        same kind of error. Only called once every open strategy has failed,
        so files that open without a recognized extension are unaffected.
        """
        path = self._error_path()
        # Only a file's extension says anything about which readers apply; a
        # directory has none, or worse, a dot in its name.
        extension = (
            (path.suffix.lstrip(".").lower() or None)
            if path is not None and not path.is_dir()
            else None
        )

        supported_extensions = get_valid_extensions()

        if extension is None or extension in supported_extensions:
            return None

        unavailable = get_unavailable_extensions()
        missing = (
            {extension: unavailable[extension]} if extension in unavailable else {}
        )
        message = (
            f"Could not convert {path}: no reader supports the .{extension} format. "
            + f"The following extensions are supported: {sorted(supported_extensions)}"
        )
        if missing:
            message += describe_missing_extras(missing)
        return UnsupportedImageFormatError(
            message,
            path=path,
            file_extension=extension,
            supported_file_extensions=tuple(sorted(supported_extensions)),
            missing_extras=tuple(sorted(set(missing.values()))),
        )

    def _classify_read_failure(self) -> CorruptImageError | None:
        """
        Decides whether a failure to open a file of a *supported* format means
        the file itself is unreadable — damaged or incompletely uploaded
        — as opposed to some other failure this class has no specific enough
        evidence to explain.
        """
        path = self._error_path()
        if path is None:
            return None
        if any(char in str(path) for char in "*?["):
            # A glob pattern rather than a file: no single file's contents to
            # blame, and the pattern may not have matched anything at all.
            return None
        if not path.is_file():
            # A missing path keeps its FileNotFoundError; a directory that
            # yielded nothing readable is not a corrupt file.
            return None
        try:
            with path.open("rb") as file:
                file.read(1)
        except OSError:
            # Not readable at all (permissions, IO error, …); that error is the
            # honest one and is already among the collected exceptions.
            return None

        return CorruptImageError(
            f"Could not read {path}. Its format is supported, so the file is "
            + "likely damaged or was not written completely.",
            path=path,
        )

    def _disable_pil_image_size_limit(self) -> None:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None

    def _try_open_slice_reader(
        self, original_images: str | list[str], exceptions: list[Exception]
    ) -> SliceReader | None:
        # try the registered reader for this extension
        def strategy_0() -> SliceReader | None:
            if isinstance(original_images, list):
                return None
            return open_slice_reader(original_images)

        # try MultiImageSliceReader, which handles a directory, glob or list of 2D images
        strategy_1 = lambda: MultiImageSliceReader(original_images)  # noqa: E731 Do not assign a `lambda` expression, use a `def`

        # for image lists, try to guess the correct reader using only the first image,
        # and apply that for all images via StackedFileSliceReader
        def strategy_2() -> SliceReader | None:
            if isinstance(original_images, list):
                # assuming the same reader works for all images:
                first_image_handler = open_slice_reader(original_images[0])
                return StackedFileSliceReader(
                    original_images, type(first_image_handler)
                )
            else:
                return None

        self._disable_pil_image_size_limit()

        for strategy in [strategy_0, strategy_1, strategy_2]:
            try:
                images_context_manager = strategy()
            except Exception as e:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
                exceptions.append(e)
            else:
                if images_context_manager is not None:
                    return images_context_manager
        return None

    @contextmanager
    def _open_slice_reader(self) -> Generator[SliceReader]:
        """
        Yields the opened reader, configured to produce slices of the form
        (self._iter_axes, *self._bundle_axes) once those are known. Before
        that, the reader is yielded with its own defaults, which is how those
        axes are found in the first place.
        """
        exceptions: list[Exception] = []
        original_images = self._resolve_original_image_paths()

        images_context_manager = self._try_open_slice_reader(
            original_images, exceptions
        )

        if images_context_manager is None:
            first_exception = exceptions[0] if exceptions else None
            if (unsupported_error := self._classify_open_failure()) is not None:
                raise unsupported_error from first_exception
            if (corrupt_error := self._classify_read_failure()) is not None:
                raise corrupt_error from first_exception
            if len(exceptions) == 1:
                raise exceptions[0]
            else:
                exceptions_str = "\n".join(
                    f"{type(e).__name__}: {str(e)}" for e in exceptions
                )
                raise ValueError(
                    f"Tried to open the images {self._original_images} with different methods, "
                    + f"none succeeded. The following errors were raised:\n{exceptions_str}"
                )

        with images_context_manager as images:
            if hasattr(self, "_bundle_axes"):
                # The axes have been worked out.
                images.default_coords.update(self._default_coords)
                images.bundle_axes = self._bundle_axes
                images.iter_axes = self._iter_axes
            yield images

    def copy_chunk_to_view(
        self,
        bbox: NormalizedBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """Copies the slices covering `bbox` into `mag_view`, one batch of
        z-slices per call. The x/y extent returned is the *observed* one,
        since expected_bbox was only a placeholder.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*is not aligned with the shard shape.*",
                category=UserWarning,
                module="webknossos",
            )
            absolute_bbox = bbox
            relative_bbox = absolute_bbox.offset(-mag_view.bounding_box.topleft)

            assert all(
                size == 1
                for size, axis in zip(absolute_bbox.size, absolute_bbox.axes)
                if axis not in CXYZ_AXES
            ), (
                "The delivered BoundingBox has to be flat except for x,y and z dimension."
            )

            # z_start and z_end are relative to the bounding box of the mag_view
            # to access the correct data from the images
            z_start, z_end = relative_bbox.get_bounds(Z_AXIS)
            shapes = []
            max_value = 0

            with self._open_slice_reader() as images:
                slices: SliceReader | _SlicedView = images
                if len(self._iter_axes) > 1:
                    # The sequence is a flat run over every iter axis. When the
                    # last one is genuinely "z", it is batched (multiple
                    # values per chunk) and narrowed to below; every other
                    # axis is exactly one value per chunk (see the assert
                    # above) and only selects where in the flat run this
                    # chunk starts. Without a real "z" — the box's "z" is then
                    # a singleton placeholder, not tied to any iterated axis —
                    # every iter axis, including the last one, is one of those
                    # single-valued selectors instead.
                    has_real_z = Z_AXIS in self._iter_axes
                    if has_real_z:
                        assert self._iter_axes[-1] == Z_AXIS, (
                            "'z' must be the last iter axis (see __init__)."
                        )
                        outer_axes = self._iter_axes[:-1]
                    else:
                        outer_axes = self._iter_axes
                    lower_bounds = images.flat_index(
                        {axis: relative_bbox.get_bounds(axis)[0] for axis in outer_axes}
                    )
                    run_length = (
                        mag_view.bounding_box.get_shape(Z_AXIS) if has_real_z else 1
                    )
                    upper_bounds = lower_bounds + run_length
                    slices = images[lower_bounds:upper_bounds]
                if self._options.flip_z:
                    slices = slices[::-1]

                with mag_view.get_buffered_slice_writer(
                    absolute_bounding_box=absolute_bbox,
                    buffer_size=absolute_bbox.get_shape(Z_AXIS),
                    allow_unaligned=True,
                ) as writer:
                    for image_slice in slices[z_start:z_end]:
                        image_slice = np.array(image_slice)
                        # place channels first
                        if C_AXIS in self._bundle_axes:
                            image_slice = np.moveaxis(
                                image_slice,
                                source=self._bundle_axes.index(C_AXIS),
                                destination=0,
                            )
                            if self._channel is not None:
                                image_slice = image_slice[
                                    self._channel : self._channel + 1
                                ]
                            elif self._first_n_channels is not None:
                                image_slice = image_slice[: self._first_n_channels]
                            assert image_slice.shape[0] == self.num_channels, (
                                f"Image shape {image_slice.shape} does not fit to the number of channels "
                                + f"{self.num_channels} which are expected in the first axis."
                            )

                        if self._options.flip_x:
                            image_slice = np.flip(image_slice, -2)
                        if self._options.flip_y:
                            image_slice = np.flip(image_slice, -1)

                        if dtype is not None:
                            image_slice = image_slice.astype(dtype, order="F")

                        max_value = max(max_value, image_slice.max())
                        if self._options.swap_xy is False:
                            image_slice = np.moveaxis(image_slice, -1, -2)

                        shapes.append(image_slice.shape[-2:])
                        writer.send(image_slice)

                return ChunkResult(dimwise_max(shapes), max_value)

    def get_layer_split_options(self) -> dict["str", list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        else:
            return self._possible_layers

    @property
    def channel(self) -> int | None:
        """The selected channel, or None if all are used. May differ from what
        was requested — a two-channel source pins channel 0 by itself."""
        return self._channel

    @property
    def expected_bbox(self) -> NormalizedBoundingBox:
        """The extents the reader reports. Only x/y is a placeholder — it is
        one slice's extent, which a later slice may exceed; the axes stepped
        through are exact, since the reader counted them.

        Always carries explicit x, y, z and "c" axes (sized `num_channels`).
        Every other axis ("t", "s", ...) is left as-is; a missing "z" gets a
        size-1 axis instead of relabeling a real one."""
        with self._open_slice_reader() as images:
            sizes = images.sizes
            x_size, y_size = sizes[X_AXIS], sizes[Y_AXIS]
            if self._options.swap_xy:
                x_size, y_size = y_size, x_size

            if len(self._iter_axes) <= 1:
                # One axis at most, so it is the z of a plain 3D box —
                # whatever the reader happens to call it.
                z_size = sizes[self._iter_axes[0]] if self._iter_axes else 1
                return BoundingBox((0, 0, 0), (x_size, y_size, z_size)).normalize_axes(
                    self.num_channels
                )

            # Several axes are stepped through (e.g. "t" and "z"), so each one
            # has to be named in the box.
            axes_names = self._iter_axes + self._bundle_axes
            axes_sizes = [sizes[axis] for axis in axes_names]
            if Z_AXIS not in axes_names:
                # No axis is genuinely called "z" (e.g. only "t" and "s" are
                # stepped through). A singleton "z" is added.
                insert_at = len(self._iter_axes)
                axes_names = axes_names[:insert_at] + [Z_AXIS] + axes_names[insert_at:]
                axes_sizes = axes_sizes[:insert_at] + [1] + axes_sizes[insert_at:]
            axes_sizes[axes_names.index(X_AXIS)] = x_size
            axes_sizes[axes_names.index(Y_AXIS)] = y_size
            if C_AXIS in axes_names:
                # sizes[C_AXIS] is the source's raw channel count, but only
                # self.num_channels of them are actually written (a pinned
                # `channel` selects one, and _first_n_channels truncates to the
                # first three).
                axes_sizes[axes_names.index(C_AXIS)] = self.num_channels
            box = NDBoundingBox.from_axes(axes_names, axes_sizes)
            return with_explicit_channel_axis(box, self.num_channels)

    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NormalizedBoundingBox
    ) -> NormalizedBoundingBox:
        """Deliberately oversized in x/y, since a write outside the layer's
        box would be rejected. Shrunk to the true extent afterwards."""
        safe_size = mag1_expected_bbox.size.with_replaced(
            mag1_expected_bbox.axes.index(X_AXIS), SAFE_LARGE_XY
        ).with_replaced(mag1_expected_bbox.axes.index(Y_AXIS), SAFE_LARGE_XY)
        return mag1_expected_bbox.with_size(safe_size)

    def chunk_grid(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NormalizedBoundingBox]:
        """Batches of z-slices spanning the full x/y extent, which cannot be
        split while it is still the placeholder."""
        del mag
        batch_size = self._resolve_batch_size(mag_view, batch_size)
        return list(
            layer_bounding_box.chunk(
                layer_bounding_box.size_xyz.with_z(batch_size),
                Vec3Int(1, 1, batch_size),
            )
        )

    @staticmethod
    def _resolve_batch_size(mag_view: MagView, batch_size: int | None) -> int:
        """Batches must not straddle whatever unit two jobs could write at
        once, or parallel writes corrupt each other: a shard when the data is
        compressed or zarr, a chunk otherwise."""
        shard_aligned = mag_view.info.compression_mode or mag_view.info.data_format in (
            DataFormat.Zarr3,
            DataFormat.Zarr,
        )
        unit = (
            mag_view.info.shard_shape.z
            if shard_aligned
            else mag_view.info.chunk_shape.z
        )
        if batch_size is None:
            return unit
        assert batch_size % unit == 0, (
            f"batch_size {batch_size} must be divisible by z "
            + f"{'shard' if shard_aligned else 'chunk'}-size {unit}"
            + (" when creating compressed layers" if shard_aligned else "")
        )
        return batch_size

    def final_bounding_box(
        self,
        layer_bounding_box: NormalizedBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NormalizedBoundingBox:
        """Replaces the placeholder x/y with the largest extent any chunk
        actually wrote."""
        return layer_bounding_box.with_size_xyz(
            Vec3Int(dimwise_max(chunk_sizes) + (layer_bounding_box.get_shape(Z_AXIS),))
            * mag.to_vec3_int().with_z(1)
        )


T = TypeVar("T", bound=tuple[int, ...])


def dimwise_max(vectors: Sequence[T]) -> T:
    if len(vectors) == 1:
        return vectors[0]
    else:
        return cast(T, tuple(map(max, *vectors)))

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import numpy as np
from natsort import natsorted
from numpy.typing import DTypeLike
from upath import UPath

from ...dataset_properties import DataFormat
from ...geometry.bounding_box import BoundingBox
from ...geometry.mag import Mag
from ...geometry.nd_bounding_box import NDBoundingBox
from ...geometry.vec3_int import Vec3Int
from ...geometry.vec_int import VecInt
from ..errors import (
    CorruptImageError,
    UnsupportedImageDataError,
    UnsupportedImageFormatError,
)
from ..layer.view import MagView
from .image_source import (
    ChunkResult,
    ImageSource,
    ReadOptions,
    compute_channel_selection,
)
from .image_source_registry import (
    describe_missing_extras,
    get_unavailable_suffixes,
    get_valid_slice_reader_suffixes,
    get_valid_suffixes,
    open_images,
)
from .raster_slices import MultiImageSlices, StackedFileSlices
from .slice_sequence import NDSliceSequence, SliceSequence

# The x/y extent is only discovered while reading, so the layer starts out
# deliberately oversized and is cut back down by final_bounding_box().
SAFE_LARGE_XY: int = 10_000_000_000  # 10 billion


def _assume_color_channel(dim_size: int, dtype: np.dtype) -> bool:
    return dim_size == 1 or (dim_size == 3 and dtype == np.dtype("uint8"))


class SlicedImageSource(ImageSource):
    # Set partway through __init__(); _open_images() uses hasattr() on
    # _bundle_axes to tell whether that has happened yet, so these must stay
    # bare annotations rather than assignments.
    _bundle_axes: list[str]
    _default_coords: dict[str, int]

    def __init__(
        self,
        image_paths: UPath | list[UPath],
        options: ReadOptions,
    ) -> None:
        """
        During initialization the readers are examined and configured to produce
        ndarrays that follow the following form:
        (self._iter_axes, *self._bundle_axis)
        self._iter_axes can be a list of different axes or an empty list if the image is 2D.
        In the latter case, the inner 2D image is still wrapped in a single-element list
        by _open_images() to be consistent with 3D images.
        self._bundle_axis can consist of "x", "y" and "c", where "c" is optional and must be
        at the start or the end, so one of "xy", "yx", "xyc", "yxc", "cxy", "cyx".

        The part "IDENTIFY AXIS ORDER" figures out (self._iter_dim, *self._img_dims)
        from the opened reader. Afterwards self._open_images() produces
        images consistent with those variables.

        The part "IDENTIFY SHAPE & CHANNELS" uses this information and the well-defined
        images to figure out the shape & num_channels.
        """
        # `images` below always refers to an opened reader, never to this
        # argument.
        self._original_images = image_paths
        self._options = options

        ## the requested channel; replaced below by the resolved one
        self._channel = options.channel
        # Only ever set by the 5D fallback below, which cannot expose a "t"
        # axis and therefore pins the first timepoint. Readers that name
        # their dimensions leave this None and keep "t" as an iter axis.
        self._timepoint: int | None = None

        ## attributes that will be set in __init__()
        self._iter_axes: list[str] = []
        self._iter_loop_size = None
        self._possible_layers = {}

        ## attributes only for NDSliceSequence instances:
        # _default_coords

        ## attributes that will also be set in __init__()
        # dtype
        # num_channels
        # _first_n_channels

        #######################
        # IDENTIFY AXIS ORDER #
        #######################

        with self._open_images() as images:
            assert isinstance(images, SliceSequence), (
                f"{type(images)} does not inherit from SliceSequence"
            )
            self.dtype = images.dtype

            if isinstance(images, NDSliceSequence):
                self._default_coords = {}

                # An image slice should always consist of a 2D image. If there are multiple channels
                # the data of each channel is part of the image slices. Possible shapes of an image
                # slice are (#y_shape, #x_shape), (1, #y_shape, #x_shape) or (3, #y_shape, #x_shape).
                if images.sizes.get("c", 1) > 1:
                    self._bundle_axes = ["c", "y", "x"]
                else:
                    if "c" in images.axes:
                        # When c-axis is not in _bundle_axes and _iter_axes its value at coordinate 0
                        # should be returned
                        self._default_coords["c"] = 0
                    self._bundle_axes = ["y", "x"]

                # All other axes are used to iterate over them. The last one is iterated the fastest.
                self._iter_axes = list(
                    set(images.axes).difference({*self._bundle_axes, "c", "z"})
                )
                if "z" in images.axes:
                    self._iter_axes.append("z")

                if len(self._iter_axes) > 1:
                    iter_size = 1
                    self._iter_loop_size = dict()
                    for axis, other_axis in zip(
                        self._iter_axes[-1:0:-1], self._iter_axes[-2::-1]
                    ):
                        # Creates a dict that contains the size of the loop for each axis
                        # the axes are identified by their index in the _iter_axes list
                        # the last axis is the fastest iterating axis, therefore the size of the loop
                        # for the last axis is 1. For all other axes it is the product of all previous axes sizes.
                        # self._iter_axes[-1:0:-1] is a reversed copy of self._iter_axes without the last element
                        # e.g. [1,2,3,4] -> [4,3,2]
                        # self._iter_axes[-2::-1] is a reversed copy of self._iter_axes without the first element
                        # e.g. [1,2,3,4] -> [3,2,1]
                        self._iter_loop_size[other_axis] = (
                            iter_size := iter_size * images.sizes[axis]
                        )

            else:
                # Fallback for flat readers that do not name their dimensions
                # as NDSliceSequence does:

                _allow_channels_first = not options.is_segmentation
                if isinstance(images, MultiImageSlices | StackedFileSlices):
                    _allow_channels_first = False

                if len(images.shape) == 2:
                    # Assume yx
                    self._bundle_axes = ["y", "x"]
                elif len(images.shape) == 3:
                    # Assume yxc, cyx or zyx
                    if _assume_color_channel(images.shape[2], images.dtype):
                        self._bundle_axes = ["y", "x", "c"]
                    elif images.shape[0] == 1 or (
                        _allow_channels_first
                        and _assume_color_channel(images.shape[0], images.dtype)
                    ):
                        self._bundle_axes = ["c", "y", "x"]
                    else:
                        self._bundle_axes = ["y", "x"]
                        self._iter_axes = ["z"]
                elif len(images.shape) == 4:
                    # Assume zcyx or zyxc
                    if images.shape[1] == 1 or _assume_color_channel(
                        images.shape[1], images.dtype
                    ):
                        self._bundle_axes = ["c", "y", "x"]
                    else:
                        self._bundle_axes = ["y", "x", "c"]
                    self._iter_axes = ["z"]
                elif len(images.shape) == 5:
                    # Assume tzcyx or tzyxc. This reader addresses t
                    # positionally (images[t]) and can only produce a 4D
                    # image, so t is held constant at the first timepoint.
                    if _assume_color_channel(images.shape[2], images.dtype):
                        self._bundle_axes = ["c", "y", "x"]
                    else:
                        self._bundle_axes = ["y", "x", "c"]
                    self._iter_axes = ["z"]
                    self._timepoint = 0
                    if images.shape[0] > 1:
                        # This fallback addresses the time dimension positionally
                        # (images[t] below) and cannot expose it as a "t" axis,
                        # so only the first timepoint is converted. Timepoints
                        # are no longer offered as a layer split — readers that
                        # name their dimensions produce a real "t" axis covering
                        # all of them in a single layer instead.
                        warnings.warn(
                            "[WARNING] The image has multiple timepoints, but this reader "
                            + "cannot address them individually, so only the first one is "
                            + "converted.",
                            UserWarning,
                        )
                else:
                    raise UnsupportedImageDataError(
                        f"Got {len(images.shape)} axes for the images, "
                        + "but don't have axes information.",
                        path=self._error_path(),
                    )

        #########################
        # IDENTIFY NUM_CHANNELS #
        #########################

        with self._open_images() as images:
            if "c" in self._bundle_axes:
                if isinstance(images, NDSliceSequence):
                    self.num_channels = images.sizes.get("c", 1)
                elif isinstance(images, list):
                    self.num_channels = cast(SliceSequence, images[0]).shape[
                        self._bundle_axes.index("c")
                    ]
                else:
                    self.num_channels = images.shape[self._bundle_axes.index("c") + 1]
            else:
                self.num_channels = 1

        self.num_channels, self._channel, self._first_n_channels, possible_channels = (
            compute_channel_selection(self.num_channels, self._channel)
        )
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

    def _normalize_original_images(self) -> str | list[str]:
        original_images = self._original_images
        if isinstance(original_images, list):
            return [str(i) for i in original_images]
        if original_images.is_dir():
            valid_suffixes = get_valid_slice_reader_suffixes()
            files: list[str] = natsorted(
                str(i)
                for i in original_images.glob("**/*")
                if i.is_file() and i.suffix.lstrip(".") in valid_suffixes
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

        The decision is made on the suffix, not on the exception type:
        open_images raises UnknownFormatError both when no reader claims the
        suffix and when every reader that claimed it errored out, so a
        corrupt TIFF is indistinguishable from an unreadable format by type.

        Classifying afterwards, rather than rejecting unknown suffixes up
        front, keeps files that open without a recognized suffix (via
        MultiImageSlices) working as before: this only runs once every open
        strategy has already failed.
        """
        path = self._error_path()
        # Only a file's suffix says anything about which readers apply: a
        # directory of images has none (or, worse, a dot in its name), and
        # neither has a path that does not exist at all.
        suffix = (
            (path.suffix.lstrip(".").lower() or None)
            if path is not None and not path.is_dir()
            else None
        )

        supported_suffixes = get_valid_suffixes()

        if suffix is None or suffix in supported_suffixes:
            return None

        unavailable = get_unavailable_suffixes()
        missing = {suffix: unavailable[suffix]} if suffix in unavailable else {}
        message = (
            f"Could not convert {path}: no reader supports the .{suffix} format. "
            + f"The following suffixes are supported: {sorted(supported_suffixes)}"
        )
        if missing:
            message += describe_missing_extras(missing)
        return UnsupportedImageFormatError(
            message,
            path=path,
            suffix=suffix,
            supported_suffixes=tuple(sorted(supported_suffixes)),
            missing_extras=tuple(sorted(set(missing.values()))),
        )

    def _classify_read_failure(self) -> CorruptImageError | None:
        """
        Decides whether a failure to open a file of a *supported* format means
        the file itself is unreadable — damaged or incompletely uploaded, by
        far the most common cause — as opposed to a problem this class should
        not put words in the user's mouth about.

        Only runs once _classify_open_failure has ruled out an unsupported
        format, so reaching here means readers exist for this suffix and none
        of them could make sense of the contents.
        """
        path = self._error_path()
        if path is None:
            return None
        if any(char in str(path) for char in "*?["):
            # A glob pattern rather than a file: no single file's contents to
            # blame, and the pattern may not have matched anything at all.
            return None
        if not path.is_file():
            # A missing path keeps its FileNotFoundError-shaped error, and a
            # directory that yielded nothing readable is not a corrupt file.
            return None
        try:
            with path.open("rb") as file:
                file.read(1)
        except OSError:
            # Not readable at all (permissions, IO error, …). That error is
            # the honest one and is already among the collected exceptions.
            return None

        return CorruptImageError(
            f"Could not read {path}. Its format is supported, so the file is "
            + "likely damaged or was not written completely.",
            path=path,
        )

    def _disable_pil_image_size_limit(self) -> None:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None

    def _try_open_images(
        self, original_images: str | list[str], exceptions: list[Exception]
    ) -> SliceSequence | None:
        # try the registered reader for this suffix
        def strategy_0() -> SliceSequence | None:
            if isinstance(original_images, list):
                return None
            return open_images(original_images)

        # try MultiImageSlices, which handles a directory, glob or list of 2D images
        strategy_1 = lambda: MultiImageSlices(original_images)  # noqa: E731 Do not assign a `lambda` expression, use a `def`

        # for image lists, try to guess the correct reader using only the first image,
        # and apply that for all images via StackedFileSlices
        def strategy_2() -> SliceSequence | None:
            if isinstance(original_images, list):
                # assuming the same reader works for all images:
                first_image_handler = open_images(original_images[0])
                return StackedFileSlices(original_images, type(first_image_handler))
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
    def _open_images(
        self,
    ) -> Iterator[Any]:
        """
        This yields well-defined images of the form (self._iter_axes, *self._bundle_axes),
        after IDENTIFY AXIS ORDER of __init__() has run.
        For a 2D image this is achieved by wrapping it in a list.

        The yielded object is a SliceSequence, a list wrapping one, or — for
        the 5D fallback that pins a timepoint — a plain ndarray. All three
        support len(), indexing and iteration, which is all callers use, but
        they share no common type; hence Any. Callers narrow with isinstance
        where they need more.
        """
        exceptions: list[Exception] = []
        original_images = self._normalize_original_images()

        images_context_manager = self._try_open_images(original_images, exceptions)

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

        with images_context_manager as opened_images:
            images: Any = opened_images
            if isinstance(images, NDSliceSequence):
                if hasattr(self, "_bundle_axes"):
                    # first part of __init__() has happened
                    images.default_coords.update(self._default_coords)
                    images.bundle_axes = self._bundle_axes
                    images.iter_axes = self._iter_axes
            else:
                if hasattr(self, "_bundle_axes"):
                    # first part of __init__() has happened
                    if self._timepoint is not None:
                        images = images[self._timepoint]
                        if "t" in self._iter_axes:
                            self._iter_axes.remove("t")
                    if not self._iter_axes:
                        # add outer list to wrap 2D images as 3D-like structure
                        images = [images]
            yield images

    def copy_chunk_to_view(
        self,
        bbox: BoundingBox | NDBoundingBox,
        mag_view: MagView,
        dtype: DTypeLike | None = None,
    ) -> ChunkResult:
        """Copies the slices covering `bbox` into `mag_view`, one batch of
        z-slices per call, meant for usage with an executor.

        Return value and axis conventions are ImageSource.copy_chunk_to_view's.
        Because expected_bbox is a placeholder for this strategy, the x/y
        extent returned here is the *observed* one, so the caller must correct
        the layer's bounding box (and largest_segment_id) afterwards.
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
                if axis not in ("c", "x", "y", "z")
            ), (
                "The delivered BoundingBox has to be flat except for x,y and z dimension."
            )

            # z_start and z_end are relative to the bounding box of the mag_view
            # to access the correct data from the images
            z_start, z_end = relative_bbox.get_bounds("z")
            shapes = []
            max_value = 0

            with self._open_images() as images:
                if self._iter_axes and self._iter_loop_size is not None:
                    # select the range of images that represents one xyz combination in the mag_view
                    lower_bounds = sum(
                        self._iter_loop_size[axis_name]
                        * relative_bbox.get_bounds(axis_name)[0]
                        for axis_name in self._iter_axes[:-1]
                    )
                    upper_bounds = lower_bounds + mag_view.bounding_box.get_shape("z")
                    images = images[lower_bounds:upper_bounds]
                if self._options.flip_z:
                    images = images[::-1]

                with mag_view.get_buffered_slice_writer(
                    # Previously only z_start and its end were important, now the slice writer needs to know
                    # which axis is currently written.
                    absolute_bounding_box=absolute_bbox,
                    buffer_size=absolute_bbox.get_shape("z"),
                    allow_unaligned=True,
                ) as writer:
                    for image_slice in images[z_start:z_end]:
                        image_slice = np.array(image_slice)
                        # place channels first
                        if "c" in self._bundle_axes:
                            image_slice = np.moveaxis(
                                image_slice,
                                source=self._bundle_axes.index("c"),
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

    def get_possible_layers(self) -> dict["str", list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        else:
            return self._possible_layers

    @property
    def channel(self) -> int | None:
        """The selected channel, or None if all channels are used.

        May differ from the `channel` constructor argument: e.g. it is set
        to 0 automatically when the source has exactly two channels.
        """
        return self._channel

    @property
    def expected_bbox(self) -> NDBoundingBox:
        # replaces the previous expected_shape to enable n-dimensional input files
        with self._open_images() as images:
            axes: Sequence[str]
            if isinstance(images, NDSliceSequence):
                axes = images.axes
                images_shape = tuple(images.sizes[axis] for axis in axes)
            else:
                if isinstance(images, list):
                    images_shape = (len(images),) + cast(SliceSequence, images[0]).shape

                else:
                    images_shape = images.shape
                if len(images_shape) == 3:
                    axes = ("z", "y", "x")
                else:
                    axes = ("z", "c", "y", "x")

            if self._iter_loop_size is None:
                # There is no or only one element in self._iter_axes, so a 3D bounding box is sufficient.
                x_index, y_index = (
                    axes.index("x"),
                    axes.index("y"),
                )
                if self._iter_axes:
                    try:
                        # In case the naming of the third axis is not "z",
                        # it is still considered as the z-axis.
                        z_index = axes.index(self._iter_axes[0])
                    except ValueError:
                        z_index = axes.index("z")
                    z_shape = images_shape[z_index]
                else:
                    z_shape = 1
                if self._options.swap_xy:
                    x_index, y_index = y_index, x_index
                return BoundingBox(
                    (0, 0, 0),
                    (images_shape[x_index], images_shape[y_index], z_shape),
                )
            else:
                if isinstance(images, NDSliceSequence):
                    axes_names = (self._iter_axes or []) + [
                        axis for axis in self._bundle_axes
                    ]
                    axes_sizes = [images.sizes[axis] for axis in axes_names]
                    if "c" in axes_names:
                        # images.sizes["c"] is the source's raw channel count,
                        # but only self.num_channels of them are actually
                        # written (a pinned `channel` selects one, and
                        # _first_n_channels truncates to the first three).
                        # Reporting the raw count here would contradict the
                        # num_channels passed to add_layer(), which surfaces as
                        # a spurious "images are larger than expected" warning
                        # and a bounding box that disagrees with the data.
                        axes_sizes[axes_names.index("c")] = self.num_channels
                    axes_index = list(range(0, len(axes_names)))
                    topleft = VecInt.zeros(tuple(axes_names))

                    if self._options.swap_xy:
                        x_index, y_index = axes_names.index("x"), axes_names.index("y")
                        axes_sizes[x_index], axes_sizes[y_index] = (
                            axes_sizes[y_index],
                            axes_sizes[x_index],
                        )

                    return NDBoundingBox(
                        topleft,
                        VecInt(axes_sizes, axes=axes_names),
                        axes_names,
                        VecInt(axes_index, axes=axes_names),
                    )

                raise UnsupportedImageDataError(
                    "It seems as if you try to load an N-dimensional image from 2D images. This is currently not supported."
                )

    def initial_layer_bounding_box(
        self, mag1_expected_bbox: NDBoundingBox
    ) -> NDBoundingBox:
        """Deliberately oversized in x/y. Reading slice by slice cannot know
        the true extent up front, and a write outside the layer's box would be
        rejected, so the box starts far too large and final_bounding_box()
        shrinks it to what was actually written."""
        safe_size = mag1_expected_bbox.size.with_replaced(
            mag1_expected_bbox.axes.index("x"), SAFE_LARGE_XY
        ).with_replaced(mag1_expected_bbox.axes.index("y"), SAFE_LARGE_XY)
        return mag1_expected_bbox.with_size(safe_size)

    def chunk_grid(
        self,
        layer_bounding_box: NDBoundingBox,
        *,
        mag_view: MagView,
        mag: Mag,
        batch_size: int | None,
    ) -> list[NDBoundingBox]:
        """Batches of z-slices spanning the full x/y extent. x/y cannot be
        split, since at this point it is still the placeholder rather than a
        real size."""
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
        once, or parallel writes corrupt each other: a whole shard when the
        data is compressed or zarr, a chunk otherwise."""
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
        layer_bounding_box: NDBoundingBox,
        *,
        chunk_sizes: Sequence[tuple[int, int]],
        mag: Mag,
    ) -> NDBoundingBox:
        """Replaces the placeholder x/y with the largest extent any chunk
        actually wrote."""
        return layer_bounding_box.with_size_xyz(
            Vec3Int(dimwise_max(chunk_sizes) + (layer_bounding_box.get_shape("z"),))
            * mag.to_vec3_int().with_z(1)
        )


T = TypeVar("T", bound=tuple[int, ...])


def dimwise_max(vectors: Sequence[T]) -> T:
    if len(vectors) == 1:
        return vectors[0]
    else:
        return cast(T, tuple(map(max, *vectors)))

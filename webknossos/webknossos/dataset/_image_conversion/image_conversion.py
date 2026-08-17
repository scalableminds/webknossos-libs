from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from enum import Enum, unique
from itertools import product
from os import PathLike
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import attr
import numpy as np
from cluster_tools import Executor
from natsort import natsort_keygen
from numpy.typing import DTypeLike
from upath import UPath

from ...dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
    LayerCategoryType,
    LayerProperties,
    SegmentationLayerProperties,
    VoxelSize,
)
from ...geometry import (
    BoundingBox,
    Mag,
    NDBoundingBox,
    Vec3FloatLike,
    Vec3Int,
    Vec3IntLike,
    VecIntLike,
)
from ...geometry.mag import MagLike
from ...utils import named_partial, wait_and_ensure_success, wrap_executor
from ..defaults import DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES, DEFAULT_DATA_FORMAT
from ..errors import UnsupportedImageDataError, UnsupportedImageFormatError
from ..layer import Layer, SegmentationLayer
from ..layer.abstract_layer import (
    _UNALLOWED_LAYER_NAME_CHARS,
    _validate_layer_name,
    channels_fit_one_layer,
)
from ..layer.layer import _get_shard_and_chunk_shapes
from . import pims_images
from .chunked_images import (
    ChunkedImages,
    describe_missing_extras,
    get_unavailable_chunked_image_extensions,
    get_valid_chunked_image_extensions,
    try_open_chunked_images,
)
from .segmentation_recognition import (
    guess_category_from_view,
    guess_if_segmentation_path,
)

if TYPE_CHECKING:
    import pims

    from ..dataset import Dataset
    from ..layer.view import MagView


logger = logging.getLogger(__name__)

SAFE_LARGE_XY: int = 10_000_000_000  # 10 billion


@unique
class ConversionLayerMapping(Enum):
    """Strategies for mapping file paths to layers when importing images.

    These strategies determine how input image files are grouped into layers during
    dataset creation using `Dataset.from_images()`. If no strategy is provided,
    `INSPECT_SINGLE_FILE` is used as the default.

    If none of the pre-defined strategies fit your needs, you can provide a custom
    callable that takes a Path and returns a layer name string.

    Examples:
        Using default strategy:
            ```
            ds = Dataset.from_images("images/", "dataset/")
            ```

        Explicit strategy:
            ```
            ds = Dataset.from_images(
                "images/",
                "dataset/",
                map_filepath_to_layer_name=ConversionLayerMapping.ENFORCE_SINGLE_LAYER
            )
            ```

        Custom mapping function:
            ```
            ds = Dataset.from_images(
                "images/",
                "dataset/",
                map_filepath_to_layer_name=lambda p: p.stem
            )
            ```
    """

    INSPECT_SINGLE_FILE = "inspect_single_file"
    """Default strategy. Inspects first image file to determine if data is 2D or 3D.
    For 2D data uses ENFORCE_LAYER_PER_FOLDER, for 3D uses ENFORCE_LAYER_PER_FILE."""

    INSPECT_EVERY_FILE = "inspect_every_file"
    """Like INSPECT_SINGLE_FILE but determines strategy separately for each file.
    More flexible but slower for many files."""

    ENFORCE_LAYER_PER_FILE = "enforce_layer_per_file"
    """Creates a new layer for each input file. Useful for converting multiple
    3D images or when each 2D image should become its own layer."""

    ENFORCE_SINGLE_LAYER = "enforce_single_layer"
    """Combines all input files into a single layer. Only useful when all
    images are 2D slices that should be combined."""

    ENFORCE_LAYER_PER_FOLDER = "enforce_layer_per_folder"
    """Groups files by their containing folder. Each folder becomes one layer.
    Useful for organized 2D image stacks."""

    ENFORCE_LAYER_PER_TOPLEVEL_FOLDER = "enforce_layer_per_toplevel_folder"
    """Groups files by their top-level folder. Useful when multiple layers each
    have their stacks split across subfolders."""

    def _to_callable(
        self,
        input_path: UPath,
        input_files: Sequence[UPath],
    ) -> Callable[[UPath], str]:
        if self == ConversionLayerMapping.ENFORCE_LAYER_PER_FILE:
            return lambda p: p.as_posix().replace("/", "_")
        elif self == ConversionLayerMapping.ENFORCE_SINGLE_LAYER:
            return lambda _p: input_path.name
        elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_FOLDER:
            return lambda p: (
                input_path.name
                if p.parent == UPath()
                else p.parent.as_posix().replace("/", "_")
            )
        elif self == ConversionLayerMapping.ENFORCE_LAYER_PER_TOPLEVEL_FOLDER:
            return lambda p: input_path.name if p.parent == UPath() else p.parts[0]
        elif self == ConversionLayerMapping.INSPECT_EVERY_FILE:
            # If a file has z dimensions, it becomes its own layer,
            # if it's 2D, the folder becomes a layer.
            return lambda p: (
                str(p)
                if _has_image_z_dimension(
                    input_path / p,
                    is_segmentation=guess_if_segmentation_path(p),
                )
                else (
                    input_path.name
                    if p.parent == UPath()
                    else p.parent.as_posix().replace("/", "_")
                )
            )
        elif self == ConversionLayerMapping.INSPECT_SINGLE_FILE:
            # As before, but only a single image is inspected to determine 2D vs 3D.
            if _has_image_z_dimension(
                input_path / input_files[0],
                is_segmentation=guess_if_segmentation_path(input_files[0]),
            ):
                return str
            else:
                return lambda p: input_path.name if p.parent == UPath() else p.parts[-2]
        else:
            raise ValueError(f"Got unexpected ConversionLayerMapping value: {self}")


def _find_unavailable_input_formats(input_upath: UPath) -> dict[str, str]:
    """
    Looks for files whose format a chunk-based reader would handle if its
    optional dependency were installed, and maps their extension to the extra
    that provides it. Empty if there are none.

    Only runs on the "no supported image data" error path, so the extra
    directory scan costs nothing in the normal case.
    """
    unavailable = get_unavailable_chunked_image_extensions()
    if not unavailable:
        return {}

    if input_upath.is_file():
        candidates = [input_upath]
    else:
        candidates = [p for p in input_upath.glob("**/*") if p.is_file()]

    found: dict[str, str] = {}
    for path in candidates:
        extension = path.suffix.lstrip(".").lower()
        if extension in unavailable:
            found[extension] = unavailable[extension]
    return found


# Formats whose channels mean RGB rather than separate acquisitions: the
# everyday 2D image formats, where three uint8 channels are RGB and belong in
# one layer. Everything else — scientific formats such as TIFF, CZI, DM3/DM4,
# .ims and MRC — stores one acquisition channel per channel, which users almost
# always want as separate layers, even when there happen to be three of them.
_RGB_IMAGE_EXTENSIONS = frozenset(
    {
        "bmp",
        "gif",
        "ico",
        "j2k",
        "jp2",
        "jpeg",
        "jpg",
        "pbm",
        "pcx",
        "pgm",
        "png",
        "ppm",
        "targa",
        "tga",
        "webp",
    }
)


def _describe_rgb_formats() -> str:
    return ", ".join("." + extension for extension in sorted(_RGB_IMAGE_EXTENSIONS))


def _channels_are_one_rgb_layer(
    num_source_channels: int,
    *,
    dtype: np.dtype,
    images: UPath | pims.FramesSequence | list[UPath],
    truncate_rgba_to_rgb: bool,
) -> bool:
    """
    Whether the source's channels are RGB channels of a single layer, rather
    than separate channels that each deserve their own layer.

    This needs the *source's* channel count, not the reader's `num_channels`:
    the readers already pin or truncate channels (a two-channel source reports
    one channel, pinned to the first), which is exactly the silent discarding
    this decision exists to avoid. Callers only reach here with 2+ source
    channels — a single channel is never offered as a possible layer split in
    the first place.
    """
    path = images if isinstance(images, UPath) else None
    if path is None and isinstance(images, list) and images:
        path = images[0]
    if path is None or path.suffix.lstrip(".").lower() not in _RGB_IMAGE_EXTENSIONS:
        return False
    # An RGBA image is written as RGB with the alpha channel dropped, unless the
    # caller asked to keep every channel, in which case they become layers.
    written_channels = (
        3 if num_source_channels == 4 and truncate_rgba_to_rgb else num_source_channels
    )
    return channels_fit_one_layer(written_channels, dtype)


def _has_image_z_dimension(
    filepath: UPath,
    is_segmentation: bool,
) -> bool:
    # Formats handled by a registered ChunkedImages subclass (e.g. .ims) are
    # never probed through PimsImages/pims — they know their exact
    # expected_bbox directly, and this is the only way such files are read.
    chunked_images = try_open_chunked_images(
        filepath,
        channel=None,
        swap_xy=False,
        flip_x=False,
        flip_y=False,
        flip_z=False,
        is_segmentation=is_segmentation,
    )
    if chunked_images is not None:
        return chunked_images.expected_bbox.get_shape("z") > 1
    return pims_images.has_image_z_dimension(filepath, is_segmentation=is_segmentation)


class _ImageConversionSetup(NamedTuple):
    """Everything add_layer_from_images needs to actually run the conversion,
    computed differently for ChunkedImages vs. PimsImages sources by
    _prepare_chunked_image_conversion / _prepare_pims_image_conversion. This
    covers the parts of the two paths that differ (bounding box, mag_view,
    per-chunk function, batch_size) — the surrounding layer/mag setup and
    execution machinery stay shared in add_layer_from_images itself.
    """

    mag_view: MagView
    func_per_chunk: Callable[..., tuple[tuple[int, int], int | None]]
    batch_size: int | None


def _prepare_chunked_image_conversion(
    image_source: ChunkedImages,
    layer: Layer,
    *,
    mag: Mag,
    mag1_expected_bbox: NDBoundingBox,
    chunk_shape: Vec3Int,
    shard_shape: Vec3Int,
    compress: bool,
    current_dtype: np.dtype,
    batch_size: int | None,
) -> _ImageConversionSetup:
    # ChunkedImages formats know their exact bounding box from metadata
    # alone, so no placeholder inflation (and no post-hoc correction
    # afterward) is needed.
    layer.bounding_box = mag1_expected_bbox

    mag_view = layer.add_mag(
        mag=mag,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=compress,
    )

    func_per_chunk = named_partial(
        image_source.read_chunk,
        mag_view=mag_view,
        dtype=current_dtype,
    )

    return _ImageConversionSetup(
        mag_view=mag_view,
        func_per_chunk=func_per_chunk,
        batch_size=batch_size,  # unused for this path, passed through unchanged
    )


def _prepare_pims_image_conversion(
    image_source: pims_images.PimsImages,
    layer: Layer,
    *,
    mag: Mag,
    mag1_expected_bbox: NDBoundingBox,
    chunk_shape: Vec3Int,
    shard_shape: Vec3Int,
    compress: bool,
    current_dtype: np.dtype,
    batch_size: int | None,
) -> _ImageConversionSetup:
    # Setting a large enough bounding box, because the exact bounding box
    # cannot be know a priori all the time. It will be replaced with the
    # correct bounding box after reading through all actual images.
    safe_size = mag1_expected_bbox.size.with_replaced(
        mag1_expected_bbox.axes.index("x"), SAFE_LARGE_XY
    ).with_replaced(mag1_expected_bbox.axes.index("y"), SAFE_LARGE_XY)
    layer.bounding_box = mag1_expected_bbox.with_size(safe_size)

    mag_view = layer.add_mag(
        mag=mag,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
        compress=compress,
    )

    if batch_size is None:
        if compress or (layer.data_format in (DataFormat.Zarr3, DataFormat.Zarr)):
            # if data is compressed or dataformat is zarr, parallel write access
            # to a shard leads to corrupted data, the batch size must be aligned
            # with the shard size
            batch_size = mag_view.info.shard_shape.z
        else:
            # in uncompressed wkw only writing to the same chunk is problematic
            batch_size = mag_view.info.chunk_shape.z
    elif compress or (layer.data_format in (DataFormat.Zarr3, DataFormat.Zarr)):
        assert batch_size % mag_view.info.shard_shape.z == 0, (
            f"batch_size {batch_size} must be divisible by z shard-size {mag_view.info.shard_shape.z} when creating compressed layers"
        )
    else:
        assert batch_size % mag_view.info.chunk_shape.z == 0, (
            f"batch_size {batch_size} must be divisible by z chunk-size {mag_view.info.chunk_shape.z}"
        )

    func_per_chunk = named_partial(
        image_source.copy_to_view,
        mag_view=mag_view,
        dtype=current_dtype,
    )

    return _ImageConversionSetup(
        mag_view=mag_view,
        func_per_chunk=func_per_chunk,
        batch_size=batch_size,
    )


def from_images(
    dataset_cls: type[Dataset],
    input_path: str | PathLike | UPath,
    output_path: str | PathLike | UPath,
    voxel_size: Vec3FloatLike | None = None,
    name: str | None = None,
    *,
    map_filepath_to_layer_name: ConversionLayerMapping
    | Callable[[UPath], str] = ConversionLayerMapping.INSPECT_SINGLE_FILE,
    z_slices_sort_key: Callable[[UPath], Any] | None = None,
    voxel_size_with_unit: VoxelSize | None = None,
    layer_name: str | None = None,
    layer_category: LayerCategoryType | None = None,
    data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
    chunk_shape: Vec3IntLike | int | None = None,
    shard_shape: Vec3IntLike | int | None = None,
    chunks_per_shard: int | Vec3IntLike | None = None,
    compress: bool = True,
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    max_layers: int = 20,
    batch_size: int | None = None,
    executor: Executor | None = None,
) -> Dataset:
    """See Dataset.from_images() for the public docstring."""
    input_upath = UPath(input_path)

    valid_extensions = pims_images.get_valid_pims_extensions()
    valid_extensions.update(get_valid_chunked_image_extensions())

    if z_slices_sort_key is None:
        z_slices_sort_key = natsort_keygen()

    # input_upath is replaced with its parent directory below for a single
    # convertible file, so keep the original around to report in the error.
    original_input_upath = input_upath
    input_is_file = input_upath.is_file()
    if input_is_file:
        if input_upath.suffix.lstrip(".").lower() in valid_extensions:
            input_files = [UPath(input_upath.name)]
            input_upath = input_upath.parent
        else:
            # No reader handles this file. Leaving input_files unassigned here
            # used to raise an UnboundLocalError instead of the error below.
            input_files = []
    else:
        input_files = [
            i.relative_to(input_upath)
            for i in input_upath.glob("**/*")
            if i.is_file() and i.suffix.lstrip(".").lower() in valid_extensions
        ]

    if len(input_files) == 0:
        message = (
            "Could not find any supported image data. "
            + f"The following extensions are supported: {sorted(valid_extensions)}"
        )
        # A reader whose optional dependency is missing never registers, so its
        # formats are simply absent from the list above. Without this the only
        # clue is an import warning emitted much earlier, far from the failure.
        missing = _find_unavailable_input_formats(original_input_upath)
        if missing:
            message += describe_missing_extras(missing)
        raise UnsupportedImageFormatError(
            message,
            path=original_input_upath,
            file_extension=(
                original_input_upath.suffix.lstrip(".").lower() or None
                if input_is_file
                else None
            ),
            supported_file_extensions=tuple(sorted(valid_extensions)),
            missing_extras=tuple(sorted(set(missing.values()))),
        )

    if isinstance(map_filepath_to_layer_name, ConversionLayerMapping):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="pims",
            )
            warnings.filterwarnings(
                "once",
                category=UserWarning,
                module="pims_images",
            )
            map_filepath_to_layer_name_func = map_filepath_to_layer_name._to_callable(
                input_upath,
                input_files=input_files,
            )
    else:
        map_filepath_to_layer_name_func = map_filepath_to_layer_name
    if voxel_size_with_unit is None:
        assert voxel_size is not None, (
            "Please supply either voxel_size or voxel_size_with_unit."
        )
        voxel_size_with_unit = VoxelSize(voxel_size)
    else:
        assert voxel_size is None, (
            "Please supply either voxel_size or voxel_size_with_unit not both."
        )

    ds = dataset_cls(output_path, voxel_size_with_unit=voxel_size_with_unit, name=name)

    filepaths_per_layer: dict[str, list[UPath]] = {}
    for input_file in input_files:
        layer_name_from_mapping = map_filepath_to_layer_name_func(input_file)
        # Remove characters from layer name that are not allowed
        layer_name_from_mapping = _UNALLOWED_LAYER_NAME_CHARS.sub(
            "", layer_name_from_mapping
        )
        # Ensure layer name does not start with a dot
        layer_name_from_mapping = layer_name_from_mapping.lstrip(".")

        assert layer_name_from_mapping != "", (
            f"Could not determine a layer name for {input_file}."
        )

        filepaths_per_layer.setdefault(layer_name_from_mapping, []).append(
            input_upath / input_file
        )

    if layer_name is not None:
        if len(filepaths_per_layer) == 1:
            filepaths_per_layer[layer_name] = filepaths_per_layer.pop(
                layer_name_from_mapping
            )
        else:
            filepaths_per_layer = {
                f"{layer_name}_{k}": v for k, v in filepaths_per_layer.items()
            }
    with wrap_executor(executor) as executor:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="pims_images",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="pims",
            )
            for layer_name, filepaths in filepaths_per_layer.items():
                filepaths.sort(key=z_slices_sort_key)

                ds.add_layer_from_images(
                    filepaths[0] if len(filepaths) == 1 else filepaths,
                    layer_name,
                    category=layer_category,
                    data_format=data_format,
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    chunks_per_shard=chunks_per_shard,
                    compress=compress,
                    swap_xy=swap_xy,
                    flip_x=flip_x,
                    flip_y=flip_y,
                    flip_z=flip_z,
                    batch_size=batch_size,
                    allow_multiple_layers=True,
                    max_layers=max_layers - len(ds.layers),
                    # An RGBA screenshot should convert into one RGB layer, not
                    # into four grayscale ones. Non-RGB channels are split by
                    # allow_multiple_layers above regardless.
                    truncate_rgba_to_rgb=True,
                    executor=executor,
                )

    return ds


def _normalize_images_argument(
    images: str | PathLike | UPath | pims.FramesSequence | list[str | PathLike | UPath],
) -> UPath | pims.FramesSequence | list[UPath]:
    """Converts every path-shaped input to UPath, so the readers only ever see
    one path type. pims.FramesSequence instances are returned unchanged."""
    if isinstance(images, (str, PathLike, UPath)):
        return UPath(images)
    if isinstance(images, list):
        return [UPath(i) for i in images]
    return images


def add_layer_from_images(
    dataset: Dataset,
    images: str | PathLike | UPath | pims.FramesSequence | list[str | PathLike | UPath],
    ## add_layer arguments
    layer_name: str,
    category: LayerCategoryType | None = "color",
    *,
    data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
    ## add_mag arguments
    mag: MagLike = Mag(1),
    chunk_shape: Vec3IntLike | int | None = None,
    shard_shape: Vec3IntLike | int | None = None,
    chunks_per_shard: int | Vec3IntLike | None = None,
    compress: bool = True,
    ## other arguments
    topleft: VecIntLike = Vec3Int.zeros(),  # in Mag(1)
    swap_xy: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    dtype: DTypeLike | None = None,
    channel: int | None = None,
    czi_channel: int | None = None,
    batch_size: int | None = None,  # defaults to shard-size z
    allow_multiple_layers: bool = False,
    max_layers: int = 20,
    truncate_rgba_to_rgb: bool = True,
    executor: Executor | None = None,
) -> Layer:
    """See Dataset.add_layer_from_images() for the public docstring."""
    _validate_layer_name(layer_name)
    # Normalize paths to UPath once, here at the boundary, so everything
    # downstream (PimsImages, ChunkedImages, try_open_chunked_images) deals in
    # a single path type and can use is_remote_path/is_fs_path directly. str
    # round-trips through UPath unchanged, including glob patterns and URLs.
    # pims.FramesSequence instances are passed through untouched.
    images = _normalize_images_argument(images)
    if category is None:
        image_path_for_category_guess = (
            images if isinstance(images, UPath) else UPath(images[0])
        )
        category = (
            "segmentation"
            if guess_if_segmentation_path(image_path_for_category_guess)
            else "color"
        )
        user_set_category = False
    else:
        user_set_category = True

    def _open_chunked_images(open_kwargs: dict) -> ChunkedImages | None:
        # Chunk-based formats (currently only .ims) know their exact
        # bounding box from metadata alone and read/write whole
        # shard-sized blocks directly.
        return try_open_chunked_images(
            images,
            channel=open_kwargs.get("channel", channel),
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            is_segmentation=category == "segmentation",
        )

    def _reopen_image_source(
        open_kwargs: dict,
    ) -> pims_images.PimsImages | ChunkedImages:
        chunked = _open_chunked_images(open_kwargs)
        if chunked is not None:
            return chunked
        return pims_images.PimsImages(
            images,
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            is_segmentation=category == "segmentation",
            **open_kwargs,
        )

    image_source: pims_images.PimsImages | ChunkedImages
    chunked_image_source = _open_chunked_images({"channel": channel})
    if chunked_image_source is not None:
        image_source = chunked_image_source
    else:
        image_source = pims_images.PimsImages(
            images,
            channel=channel,
            czi_channel=czi_channel,
            swap_xy=swap_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            is_segmentation=category == "segmentation",
        )
    possible_layers = image_source.get_possible_layers()
    # The dtype the layer will be written with decides whether its channels can
    # share one layer, so an explicit `dtype` argument counts here too.
    layer_dtype = np.dtype(dtype) if dtype is not None else np.dtype(image_source.dtype)
    source_channels = (
        possible_layers.get("channel") if possible_layers is not None else None
    )
    if source_channels is not None:
        if _channels_are_one_rgb_layer(
            len(source_channels),
            dtype=layer_dtype,
            images=images,
            truncate_rgba_to_rgb=truncate_rgba_to_rgb,
        ):
            # The channels are RGB, so they belong in one layer rather than
            # being offered up as one layer each. The readers take care of the
            # RGBA -> RGB truncation.
            assert possible_layers is not None
            del possible_layers["channel"]
        elif not allow_multiple_layers:
            raise UnsupportedImageDataError(
                f"Cannot write these {len(source_channels)} channels into a "
                + "single layer: WEBKNOSSOS only combines several channels into "
                + f"one layer as RGB (three uint8 channels of {_describe_rgb_formats()}). "
                + "Set allow_multiple_layers=True to write one layer per channel, "
                + "or channel=<index> to convert a single channel.",
                path=images if isinstance(images, UPath) else None,
            )
    # Further below, we iterate over suffix_with_pims_open_kwargs_per_layer in the for-loop
    # to add one layer per possible_layer if allow_multiple_layers is True.
    # If just a single layer is added, we still add a default value in the dict.
    if possible_layers is not None and len(possible_layers) > 0:
        if allow_multiple_layers:
            # Get all combinations of possible layers. E.g.
            # possible_layers = {
            #    "channel": [0, 1, 2],
            #    "czi_channel": [0, 1],
            # }
            # suffix_with_pims_open_kwargs_per_layer = {
            #    "__channel0_czi_channel0", {"channel": 0, "czi_channel": 0},
            #    "__channel0_czi_channel1", {"channel": 0, "czi_channel": 1},
            #    …,
            #    "__channel1_czi_channel0", {"channel": 1, "czi_channel": 0},
            #    …,
            # }
            # Timepoints are never split this way: readers that can address
            # them expose all timepoints on a "t" axis within a single layer.
            suffix_with_pims_open_kwargs_per_layer = {
                "__" + "_".join(f"{k}{v}" for k, v in sorted(pairs)): dict(pairs)
                for pairs in product(
                    *(
                        [(key, value) for value in values]
                        for key, values in possible_layers.items()
                    )
                )
            }
        else:
            # initialize PimsImages as above, with normal layer name
            suffix_with_pims_open_kwargs_per_layer = {"": {}}
            # Channels are already resolved above — either written into this
            # layer as RGB, or refused — so anything left here is a dimension
            # that really does default to its first value.
            warnings.warn(
                f"[INFO] There are dimensions beyond channels and xyz which cannot be read: {possible_layers}. "
                "Defaulting to the first one. "
                "Please set allow_multiple_layers=True if all of them should be written to different layers, "
                "or set specific values as arguments.",
            )
    else:
        # initialize PimsImages as above, with normal layer name
        suffix_with_pims_open_kwargs_per_layer = {"": {}}
    first_layer = None
    add_layer_kwargs = {}
    if category == "segmentation":
        add_layer_kwargs["largest_segment_id"] = 0
    if len(suffix_with_pims_open_kwargs_per_layer) > max_layers:
        warnings.warn(
            f"[INFO] Limiting the number of added layers to {max_layers} out of {len(suffix_with_pims_open_kwargs_per_layer)}. "
            + "Please increase `max_layers` if you want more layers to be added.",
        )
    for _, (
        layer_name_suffix,
        pims_open_kwargs,
    ) in zip(range(max_layers), suffix_with_pims_open_kwargs_per_layer.items()):
        # If pims_open_kwargs is empty there's no need to re-open the images:
        if len(pims_open_kwargs) > 0:
            # Set parameters from this method as default
            # if they are not part of the kwargs per layer:
            pims_open_kwargs.setdefault("channel", channel)  # type: ignore
            pims_open_kwargs.setdefault("czi_channel", czi_channel)  # type: ignore
            image_source = _reopen_image_source(pims_open_kwargs)
        if dtype is None:
            current_dtype = np.dtype(image_source.dtype)
            if current_dtype.byteorder == ">":
                current_dtype = current_dtype.newbyteorder("<")
        else:
            current_dtype = np.dtype(dtype)

        try:
            layer = dataset.add_layer(
                layer_name=layer_name + layer_name_suffix,
                category=category,
                data_format=data_format,
                dtype=current_dtype,
                num_channels=image_source.num_channels,
                **add_layer_kwargs,  # type: ignore[arg-type]
            )
        except ValueError as e:
            # add_layer rejects dtypes WEBKNOSSOS cannot store (e.g. float data
            # for a segmentation layer). Here that dtype comes from the images
            # themselves, or from the caller's `dtype` argument, so it is a
            # problem with the input rather than a programming error.
            raise UnsupportedImageDataError(str(e)) from e

        expected_bbox = image_source.expected_bbox

        _shard_shape_user_specified = (
            shard_shape is not None or chunks_per_shard is not None
        )
        chunk_shape, shard_shape = _get_shard_and_chunk_shapes(
            data_format=layer.data_format,
            layer_bounding_box=None,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            shard_shape=shard_shape,
        )
        # For Zarr3 image imports, default to z-aligned shards for efficient
        # slice-by-slice writing, regardless of total z size.
        if not _shard_shape_user_specified and layer.data_format == DataFormat.Zarr3:
            shard_shape = chunk_shape * DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES
        # When the expected bbox is 2D the chunk_shape is set to 2D too.
        if expected_bbox.get_shape("z") == 1 and layer.data_format == DataFormat.Zarr3:
            chunk_shape = chunk_shape.with_z(1)
            shard_shape = shard_shape.with_z(1)

        mag = Mag(mag)
        mag1_expected_bbox = expected_bbox.from_mag_to_mag1(mag).offset(topleft)

        prepare_conversion = (
            _prepare_chunked_image_conversion
            if isinstance(image_source, ChunkedImages)
            else _prepare_pims_image_conversion
        )
        mag_view, func_per_chunk, batch_size = prepare_conversion(
            image_source,  # type: ignore[arg-type]
            layer,
            mag=mag,
            mag1_expected_bbox=mag1_expected_bbox,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=compress,
            current_dtype=current_dtype,
            batch_size=batch_size,
        )

        if (
            additional_axes := set(layer.bounding_box.axes).difference(
                "c", "x", "y", "z"
            )
        ) and layer.data_format == DataFormat.WKW:
            if all(layer.bounding_box.get_shape(axis) == 1 for axis in additional_axes):
                warnings.warn(
                    f"[INFO] The data has additional axes {additional_axes}, but they are all of size 1. "
                    + "These axes are not stored in the layer."
                )
                layer.bounding_box = BoundingBox.from_ndbbox(layer.bounding_box)
            else:
                raise UnsupportedImageDataError(
                    f"WKW datasets only support x, y, z axes, got {additional_axes}. Please use `data_format='zarr3'` instead."
                )

        if isinstance(image_source, ChunkedImages):
            # Split into full 3D shard-aligned chunks so each job reads
            # exactly one shard's worth of data and writes it directly.
            # layer.bounding_box is already the exact bbox (no placeholder
            # inflation) for ChunkedImages, so chunking it directly is safe.
            # It is in Mag(1) while shard_shape is in mag space, so the shard
            # shape is scaled up to match; otherwise mag > 1 produces chunks a
            # factor of `mag` too small and read_chunk is handed bounds that
            # run past the end of the source.
            chunked_shard_shape = mag_view.info.shard_shape * mag.to_vec3_int()
            args = list(
                layer.bounding_box.chunk(chunked_shard_shape, chunked_shard_shape)
            )
        else:
            # _prepare_pims_image_conversion always resolves batch_size to
            # a concrete int; only the (unused-here) ChunkedImages path
            # leaves it as None.
            assert batch_size is not None
            buffered_slice_writer_shape = layer.bounding_box.size_xyz.with_z(batch_size)
            args = list(
                layer.bounding_box.chunk(
                    buffered_slice_writer_shape,
                    Vec3Int(1, 1, batch_size),
                )
            )

        with warnings.catch_warnings():
            # We need to catch and ignore a warning here about comparing persisted properties.
            # The problem is that there is an `axisOrder` property that goes in the datasource-properties.json
            # file, but is only stored in a mag object. However, at first we don't have any mags
            # so there is no place to store them. When we add the first mag and update the properties,
            # there is a check that reads the properties first and compares them to the current state.
            # This check fails because the axisOrder property hasn't been stored in the properties file.
            # It is safe to ignore this warning because it is only an intermediate problem in this function.
            # At the end of this function, the properties are complete and consistent.
            warnings.filterwarnings(
                "ignore",
                message=".* properties changed in a way that they are not comparable anymore. Most likely the bounding box naming or axis order changed.*",
                category=UserWarning,
                module="webknossos",
            )
            with wrap_executor(executor) as executor:
                shapes_and_max_ids = wait_and_ensure_success(
                    executor.map_to_futures(func_per_chunk, args),
                    executor=executor,
                    progress_desc=f"Creating layer [bold blue]{layer.name}[/bold blue] from images",
                )
            shapes, max_ids = zip(*shapes_and_max_ids)
            if category == "segmentation":
                max_id = max(max_ids)
                cast(SegmentationLayer, layer).largest_segment_id = max_id
            if not isinstance(image_source, ChunkedImages):
                # ChunkedImages formats already have the exact bbox set
                # above; correcting it from per-chunk shapes here would be
                # wrong for them anyway, since each job's shape reflects
                # only its own shard-sized chunk, not the full extent.
                layer.bounding_box = layer.bounding_box.with_size_xyz(
                    Vec3Int(
                        pims_images.dimwise_max(shapes)
                        + (layer.bounding_box.get_shape("z"),)
                    )
                    * mag.to_vec3_int().with_z(1)
                )
        if expected_bbox != layer.bounding_box:
            warnings.warn(
                "[WARNING] Some images are larger than expected, smaller slices are padded with zeros now. "
                + f"New bbox is {layer.bounding_box}, expected {expected_bbox}."
            )

        # Check if category of layer is set correctly
        try:
            if not user_set_category:
                # When the category is not set by the user, we use a very simple heuristic to guess the category
                # based on the file name of the input images. After loading the images, we check if the guessed
                # category might be wrong and adjust it if necessary. This second heuristic is based on the
                # pixel data of the images
                guessed_category = guess_category_from_view(layer.get_finest_mag())
                if guessed_category != layer.category:
                    new_layer_properties: LayerProperties
                    if guessed_category == SEGMENTATION_CATEGORY:
                        logger.info("The layer category is set to segmentation.")
                        new_layer_properties = SegmentationLayerProperties(
                            **(
                                attr.asdict(layer._properties, recurse=False)
                            ),  # use all attributes from LayerProperties
                            largest_segment_id=int(max(max_ids)),
                        )
                        new_layer_properties.category = SEGMENTATION_CATEGORY
                        dataset._layers[layer.name] = SegmentationLayer(
                            dataset, new_layer_properties, read_only=False
                        )
                    else:
                        logger.info("The layer category is set to color.")
                        _properties = attr.asdict(layer._properties, recurse=False)
                        _properties.pop("largest_segment_id", None)
                        _properties.pop("mappings", None)

                        new_layer_properties = LayerProperties(**_properties)
                        new_layer_properties.category = COLOR_CATEGORY
                        dataset._layers[layer.name] = Layer(
                            dataset, new_layer_properties, read_only=False
                        )
                    dataset._properties.update_for_layer(
                        layer.name, new_layer_properties
                    )
                    dataset._save_dataset_properties()

        except Exception:
            # The used heuristic was not able to guess the layer category, the previous value is kept
            pass

        # Verify the properties are actually consistent now, instead of assuming so.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=".* properties changed in a way that they are not comparable anymore. Most likely the bounding box naming or axis order changed.*",
                category=UserWarning,
                module="webknossos",
            )
            dataset._save_dataset_properties()
        if first_layer is None:
            first_layer = layer
    assert first_layer is not None
    return first_layer

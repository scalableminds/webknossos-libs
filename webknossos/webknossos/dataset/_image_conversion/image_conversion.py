"""Implementation of `Dataset.from_images()` / `Dataset.add_layer_from_images()`:
mapping input image files to layers, then converting them via an `ImageSource`.

The public functions on `Dataset` are thin wrappers around the free functions
here, keeping the public API and docstrings on the `Dataset` class itself.
"""

from __future__ import annotations

import colorsys
import logging
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from enum import Enum, unique
from itertools import product
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, cast

import attr
import numpy as np
from cluster_tools import Executor
from natsort import natsort_keygen
from numpy.typing import DTypeLike
from upath import UPath

from ... import utils
from ...dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    AffineCoordinateTransformation,
    CoordinateTransformation,
    DataFormat,
    LayerCategoryType,
    LayerProperties,
    LayerViewConfiguration,
    SegmentationLayerProperties,
    VoxelSize,
)
from ...geometry import (
    BoundingBox,
    Mag,
    Vec3FloatLike,
    Vec3Int,
    Vec3IntLike,
    VecIntLike,
)
from ...geometry.constants import CXYZ_AXES, Z_AXIS
from ...geometry.mag import MagLike
from ...utils import named_partial, wait_and_ensure_success
from ..defaults import DEFAULT_CHUNKS_PER_SHARD_FROM_IMAGES, DEFAULT_DATA_FORMAT
from ..errors import UnsupportedImageDataError, UnsupportedImageFormatError
from ..layer import Layer, SegmentationLayer
from ..layer.abstract_layer import (
    _UNALLOWED_LAYER_NAME_CHARS,
    _validate_layer_name,
    channels_fit_one_layer,
)
from ..layer.layer import _get_shard_and_chunk_shapes
from . import common_slice_readers, sliced_image_source
from .image_source import ImageSource, ReadOptions, ValueRange
from .image_source_registry import (
    describe_missing_extras,
    get_unavailable_extensions,
    get_valid_extensions,
    is_chunked_source_directory,
    open_image_source,
)
from .segmentation_recognition import (
    guess_category_from_view,
    guess_if_segmentation_path,
)

if TYPE_CHECKING:
    from ..dataset import Dataset


logger = logging.getLogger(__name__)

# Warnings like "loading ALL files in this directory" are meant for a caller
# who named one input, not for every file of a bulk conversion. Taken from the
# modules themselves because filterwarnings matches on the module name, which a
# rename would otherwise silently break.
_COMMON_SLICE_READERS_MODULE = common_slice_readers.__name__.rsplit(".", 1)[-1]
_SLICED_SOURCE_MODULE = sliced_image_source.__name__.rsplit(".", 1)[-1]


@contextmanager
def _quiet_reader_warnings(
    sliced_source_action: Literal["ignore", "once"] = "ignore",
) -> Generator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=_COMMON_SLICE_READERS_MODULE
        )
        warnings.filterwarnings(
            sliced_source_action, category=UserWarning, module=_SLICED_SOURCE_MODULE
        )
        yield


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
                if _has_image_z_dimension(input_path / p)
                else (
                    input_path.name
                    if p.parent == UPath()
                    else p.parent.as_posix().replace("/", "_")
                )
            )
        elif self == ConversionLayerMapping.INSPECT_SINGLE_FILE:
            # As before, but only a single image is inspected to determine 2D vs 3D.
            if _has_image_z_dimension(input_path / input_files[0]):
                return str
            else:
                return lambda p: input_path.name if p.parent == UPath() else p.parts[-2]
        else:
            raise ValueError(f"Got unexpected ConversionLayerMapping value: {self}")


def _iter_convertible_paths(root: UPath, valid_extensions: set[str]) -> Iterator[UPath]:
    """Walks `root`, yielding files with a supported extension and, as single
    leaf entries, directories a chunked image source recognizes as its own store
    (e.g. a Zarr/N5/neuroglancer-precomputed root) — those are never descended
    into, since their contents are the store's internal chunks, not more
    input files."""
    for child in root.iterdir():
        if child.is_dir():
            if is_chunked_source_directory(child):
                yield child
            else:
                yield from _iter_convertible_paths(child, valid_extensions)
        elif child.suffix.lstrip(".").lower() in valid_extensions:
            yield child


def _find_unavailable_input_formats(input_upath: UPath) -> dict[str, str]:
    """
    Looks for files whose format a chunk-based reader would handle if its
    optional dependency were installed, and maps their extension to the extra
    that provides it. Empty if there are none.

    Only runs on the "no supported image data" error path, so the extra
    directory scan costs nothing in the normal case.
    """
    unavailable = get_unavailable_extensions()
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
# everyday 2D image formats, where three uint8 channels are RGB and always
# belong in one layer, regardless of allow_multiple_layers (see
# SingleImageSliceReader.channels_are_rgb). Everything else — scientific
# formats such as TIFF, CZI, DM3/DM4, .ims and MRC — stores one acquisition
# channel per channel, which users may want as separate layers even when
# there happen to be three of them, so those are only combined into an RGB
# layer as a fallback when allow_multiple_layers=False; passing
# allow_multiple_layers=True still splits them.
_RGB_IMAGE_EXTENSIONS = frozenset(
    {
        "bmp",
        "gif",
        "ico",
        "jpeg",
        "jpg",
        "png",
    }
)


def _describe_rgb_formats() -> str:
    return ", ".join("." + extension for extension in sorted(_RGB_IMAGE_EXTENSIONS))


# Colors assigned to the layers created by splitting a multi-channel image
# into one layer per channel: the first three channels are so often the
# acquisition's own red/green/blue-ish stains that defaulting to actual red,
# green and blue makes the split layers overlay sensibly right away. Further
# channels get evenly spaced hues instead of a fixed color, since there is no
# similarly universal convention for a fourth, fifth, … channel.
def _channel_layer_color(channel_index: int) -> tuple[int, int, int]:
    if channel_index == 0:
        return (255, 0, 0)
    if channel_index == 1:
        return (0, 255, 0)
    if channel_index == 2:
        return (0, 0, 255)

    # The golden-angle increment spreads any number of additional channels
    # across the hue circle so consecutive ones stay visually distinct.
    hue = ((channel_index - 3) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (round(red * 255), round(green * 255), round(blue * 255))


def _channels_are_one_rgb_layer(
    num_source_channels: int,
    *,
    dtype: np.dtype,
    image_source: ImageSource,
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
    if num_source_channels == 1:
        return True
    if not image_source.channels_are_rgb:
        return False
    # An RGBA image is written as RGB with the alpha channel dropped, unless the
    # caller asked to keep every channel, in which case they become layers.
    written_channels = (
        3 if num_source_channels == 4 and truncate_rgba_to_rgb else num_source_channels
    )
    return channels_fit_one_layer(written_channels, dtype)


def _has_image_z_dimension(filepath: UPath) -> bool:
    # No option affects the z extent, so the defaults will do.
    return (
        open_image_source(filepath, ReadOptions()).expected_bbox.get_shape(Z_AXIS) > 1
    )


def _voxel_size_from_images(
    filepaths_per_layer: dict[str, list[UPath]],
) -> VoxelSize | None:
    """The voxel size the images report themselves, for callers that supplied
    none. Layers that report nothing are skipped; if they disagree, the first
    one wins. None when no layer reports one."""
    suggestions: list[tuple[str, VoxelSize]] = []
    with _quiet_reader_warnings():
        for name, filepaths in filepaths_per_layer.items():
            try:
                source = open_image_source(
                    filepaths[0] if len(filepaths) == 1 else sorted(filepaths),
                    ReadOptions(),
                )
            except Exception:
                # Opening it again below reports this properly.
                continue
            if (suggested := source.suggested_voxel_size) is not None:
                suggestions.append((name, suggested))
    if len(suggestions) == 0:
        return None
    first_name, first = suggestions[0]
    differing = [
        name
        for name, suggested in suggestions[1:]
        if not np.allclose(suggested.to_nanometer(), first.to_nanometer())
    ]
    if differing:
        warnings.warn(
            f"[INFO] The images report different voxel sizes. Using {first} from "
            f"{first_name} for the whole dataset, also for {differing}."
        )
    return first


def _in_dataset_voxels(
    transformations: tuple[CoordinateTransformation, ...],
    *,
    suggested_voxel_size: VoxelSize | None,
    dataset_voxel_size: VoxelSize,
) -> list[CoordinateTransformation]:
    """A source states its transformations in voxels of the voxel size it
    suggests, while WEBKNOSSOS reads them in the dataset's voxels. Conjugating
    by the ratio converts between the two."""
    if suggested_voxel_size is None:
        return list(transformations)
    ratio = np.asarray(suggested_voxel_size.to_nanometer()) / np.asarray(
        dataset_voxel_size.to_nanometer()
    )
    if np.allclose(ratio, 1.0):
        return list(transformations)
    scale = np.diag([*ratio, 1.0])
    return [
        AffineCoordinateTransformation(
            matrix=scale @ transformation.matrix @ np.linalg.inv(scale)
        )
        if isinstance(transformation, AffineCoordinateTransformation)
        else transformation
        for transformation in transformations
    ]


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

    valid_extensions = get_valid_extensions()

    if z_slices_sort_key is None:
        z_slices_sort_key = natsort_keygen()

    # input_upath is replaced with its parent directory below for a single
    # convertible file or store directory, so keep the original around to
    # report in the error.
    original_input_upath = input_upath
    input_is_file = input_upath.is_file()
    if input_is_file:
        if input_upath.suffix.lstrip(".").lower() in valid_extensions:
            input_files = [UPath(input_upath.name)]
            input_upath = input_upath.parent
        else:
            # No reader handles this file.
            input_files = []
    elif is_chunked_source_directory(input_upath):
        # input_upath is itself one store (e.g. a Zarr/N5/neuroglancer-
        # precomputed root), not a directory of inputs to walk — the same
        # single-entry case as a convertible file above, since
        # _iter_convertible_paths only recognizes such a store among a
        # directory's children, never the root it was called with.
        input_files = [UPath(input_upath.name)]
        input_upath = input_upath.parent
    else:
        input_files = [
            i.relative_to(input_upath)
            for i in _iter_convertible_paths(input_upath, valid_extensions)
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
        with _quiet_reader_warnings(sliced_source_action="once"):
            map_filepath_to_layer_name_func = map_filepath_to_layer_name._to_callable(
                input_upath,
                input_files=input_files,
            )
    else:
        map_filepath_to_layer_name_func = map_filepath_to_layer_name
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

    if voxel_size_with_unit is None:
        if voxel_size is None:
            voxel_size_with_unit = _voxel_size_from_images(filepaths_per_layer)
            assert voxel_size_with_unit is not None, (
                "Please supply either voxel_size or voxel_size_with_unit. The "
                "images carry no voxel size of their own."
            )
        else:
            voxel_size_with_unit = VoxelSize(voxel_size)
    else:
        assert voxel_size is None, (
            "Please supply either voxel_size or voxel_size_with_unit not both."
        )

    ds = dataset_cls(output_path, voxel_size_with_unit=voxel_size_with_unit, name=name)

    with utils.wrap_executor(executor) as executor:
        with _quiet_reader_warnings():
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
    images: str | PathLike | UPath | Sequence[str | PathLike | UPath],
) -> UPath | list[UPath]:
    """Converts every path-shaped input to UPath, so the readers only ever see
    one path type."""
    if isinstance(images, str | PathLike | UPath):
        return UPath(images)
    return [UPath(i) for i in images]


def add_layer_from_images(
    dataset: Dataset,
    images: str | PathLike | UPath | Sequence[str | PathLike | UPath],
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
    scale: int | None = None,
    batch_size: int | None = None,  # defaults to shard-size z
    allow_multiple_layers: bool = False,
    max_layers: int = 20,
    truncate_rgba_to_rgb: bool = True,
    executor: Executor | None = None,
) -> Layer:
    """See Dataset.add_layer_from_images() for the public docstring."""
    _validate_layer_name(layer_name)
    # Normalize to UPath once, here at the boundary, so everything downstream
    # deals in a single path type. str round-trips unchanged, including glob
    # patterns and URLs.
    image_paths = _normalize_images_argument(images)
    del images
    if category is None:
        image_path_for_category_guess = (
            image_paths if isinstance(image_paths, UPath) else UPath(image_paths[0])
        )
        category = (
            "segmentation"
            if guess_if_segmentation_path(image_path_for_category_guess)
            else "color"
        )
        user_set_category = False
    else:
        user_set_category = True

    # czi_channel/scale are each meaningful to some readers only; the rest
    # ignore the name.
    read_options = ReadOptions(
        channel=channel,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_z=flip_z,
        format_options={"czi_channel": czi_channel, "scale": scale},
    )

    image_source: ImageSource = open_image_source(image_paths, read_options)
    possible_layers = image_source.get_layer_split_options()
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
            image_source=image_source,
            truncate_rgba_to_rgb=truncate_rgba_to_rgb,
        ):
            # The channels are RGB, so they belong in one layer rather than
            # being offered up as one layer each. The readers take care of the
            # RGBA -> RGB truncation.
            assert possible_layers is not None
            del possible_layers["channel"]
        elif not allow_multiple_layers:
            if channels_fit_one_layer(len(source_channels), layer_dtype):
                # Not one of the RGB image formats, but three uint8 channels
                # still display as RGB in WEBKNOSSOS, so write them into one
                # layer instead of erroring, since the caller did not ask for
                # separate layers via allow_multiple_layers=True.
                assert possible_layers is not None
                del possible_layers["channel"]
            else:
                raise UnsupportedImageDataError(
                    f"Cannot write these {len(source_channels)} channels into a "
                    + "single layer: WEBKNOSSOS only combines several channels into "
                    + f"one layer as RGB (three uint8 channels of {_describe_rgb_formats()}). "
                    + "Set allow_multiple_layers=True to write one layer per channel, "
                    + "or channel=<index> to convert a single channel.",
                    path=image_paths if isinstance(image_paths, UPath) else None,
                )
    # "channel" only survives in possible_layers when its channels are being
    # split into one layer each (both branches above delete it once the
    # channels are combined into a single RGB layer instead), so this is
    # exactly the condition under which the per-channel colors below apply.
    splitting_channels_into_layers = (
        possible_layers is not None and "channel" in possible_layers
    )
    # One layer is added per entry below, named with its suffix; a single
    # entry with an empty suffix and selection means just one default layer.
    if possible_layers is not None and len(possible_layers) > 0:
        if allow_multiple_layers:
            # Every combination of possible layers, e.g. {"channel": [0, 1],
            # "czi_channel": [0, 1]} becomes suffixes "__channel0_czi_channel0",
            # "__channel0_czi_channel1", etc., each mapped to its selection.
            # Timepoints are never split this way: readers that can address
            # them expose all timepoints on a "t" axis within a single layer.
            suffix_with_open_kwargs_per_layer = {
                "__"
                + "_".join(
                    image_source.layer_split_label(k, v) for k, v in sorted(pairs)
                ): dict(pairs)
                for pairs in product(
                    *(
                        [(key, value) for value in values]
                        for key, values in possible_layers.items()
                    )
                )
            }
        else:
            # A single default layer, with the normal layer name.
            suffix_with_open_kwargs_per_layer = {"": {}}
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
        # A single default layer, with the normal layer name.
        suffix_with_open_kwargs_per_layer = {"": {}}
    first_layer = None
    add_layer_kwargs = {}
    if category == "segmentation":
        add_layer_kwargs["largest_segment_id"] = 0
    if len(suffix_with_open_kwargs_per_layer) > max_layers:
        warnings.warn(
            f"[INFO] Limiting the number of added layers to {max_layers} out of {len(suffix_with_open_kwargs_per_layer)}. "
            + "Please increase `max_layers` if you want more layers to be added.",
        )
    for _, (
        layer_name_suffix,
        layer_selection,
    ) in zip(range(max_layers), suffix_with_open_kwargs_per_layer.items()):
        # If layer_selection is empty there's no need to re-open the images:
        if len(layer_selection) > 0:
            image_source = open_image_source(
                image_paths, read_options.with_layer_selection(layer_selection)
            )
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

        if category == "color":
            # A format may suggest its own display defaults for the channel
            # this layer was written from (e.g. OME-Zarr's `omero` channel
            # metadata) — used as is when pinned to a single channel, or
            # merged with the split colors below.
            suggested = image_source.suggested_view_configuration
            if splitting_channels_into_layers:
                # layer_selection["channel"] is set for every layer here (it
                # is one of the keys that produced
                # suffix_with_open_kwargs_per_layer's combinations), pinning
                # this layer to the channel it was split off from.
                fallback_color = _channel_layer_color(layer_selection["channel"])
                color = (suggested.color if suggested else None) or fallback_color
                layer.default_view_configuration = (
                    attr.evolve(suggested, color=color)
                    if suggested is not None
                    else LayerViewConfiguration(color=color)
                )
            elif suggested is not None:
                layer.default_view_configuration = suggested

        # OME-Zarr's coordinate transformations place the data in space. Only
        # the layer's metadata records this — the data itself is written at the
        # same position it would be without them.
        if transformations := image_source.suggested_coordinate_transformations:
            layer.coordinate_transformations = _in_dataset_voxels(
                transformations,
                suggested_voxel_size=image_source.suggested_voxel_size,
                dataset_voxel_size=dataset.voxel_size_with_unit,
            )

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
        if (
            expected_bbox.get_shape(Z_AXIS) == 1
            and layer.data_format == DataFormat.Zarr3
        ):
            chunk_shape = chunk_shape.with_z(1)
            shard_shape = shard_shape.with_z(1)

        mag = Mag(mag)
        mag1_expected_bbox = expected_bbox.from_mag_to_mag1(mag).offset(topleft)

        layer.bounding_box = image_source.initial_layer_bounding_box(mag1_expected_bbox)
        mag_view = layer.add_mag(
            mag=mag,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            compress=compress,
        )
        func_per_chunk = named_partial(
            image_source.copy_chunk_to_view,
            mag_view=mag_view,
            dtype=current_dtype,
        )

        if (
            additional_axes := set(layer.bounding_box.axes).difference(*CXYZ_AXES)
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

        args = image_source.chunk_grid(
            layer.normalized_bounding_box,
            mag_view=mag_view,
            mag=mag,
            batch_size=batch_size,
        )

        with warnings.catch_warnings():
            # The layer has no mags yet, so its `axisOrder` property is not
            # yet stored; comparing against the persisted properties then
            # spuriously warns until the first mag is added below.
            warnings.filterwarnings(
                "ignore",
                message=".* properties changed in a way that they are not comparable anymore. Most likely the bounding box naming or axis order changed.*",
                category=UserWarning,
                module="webknossos",
            )
            with utils.wrap_executor(executor) as executor:
                chunk_results = wait_and_ensure_success(
                    executor.map_to_futures(func_per_chunk, args),
                    executor=executor,
                    progress_desc=f"Creating layer [bold blue]{layer.name}[/bold blue] from images",
                )
            shapes = [result.xy_size for result in chunk_results]
            max_ids = [result.max_value for result in chunk_results]
            value_range = ValueRange.combined(
                result.value_range for result in chunk_results
            )
            if category == "segmentation":
                max_id = max(max_ids)
                cast(SegmentationLayer, layer).largest_segment_id = max_id
            layer.bounding_box = image_source.final_bounding_box(
                layer.normalized_bounding_box, chunk_sizes=shapes, mag=mag
            )
        if expected_bbox != layer.bounding_box:
            warnings.warn(
                "[WARNING] Some images are larger than expected, smaller slices are padded with zeros now. "
                + f"New bbox is {layer.bounding_box}, expected {expected_bbox}."
            )

        try:
            if not user_set_category:
                # The category was only guessed from the input file names, so
                # double-check it against the actual pixel data now that it
                # has been read.
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
            # Could not guess the category; keep the one already set.
            pass

        # The category guess may have replaced the layer object.
        final_layer = dataset._layers[layer.name]
        if final_layer.category == COLOR_CATEGORY and value_range is not None:
            view_configuration = final_layer.default_view_configuration
            if view_configuration is None:
                final_layer.default_view_configuration = LayerViewConfiguration(
                    min=value_range.min, max=value_range.max
                )
            elif view_configuration.min is None or view_configuration.max is None:
                # The format's own metadata wins, so only fill the gaps.
                final_layer.default_view_configuration = attr.evolve(
                    view_configuration,
                    min=view_configuration.min
                    if view_configuration.min is not None
                    else value_range.min,
                    max=view_configuration.max
                    if view_configuration.max is not None
                    else value_range.max,
                )

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

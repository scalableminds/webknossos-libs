"""Which reader handles which file suffix, for both reading strategies.

Two kinds of class register here:

* slice readers (`SliceSequence` subclasses), opened by `open_images()` and
  driven by `SlicedImageSource`;
* `ChunkedImageSource` subclasses, opened directly by
  `try_open_chunked_image_source()`.

They are separate mechanisms — a suffix belongs to one or the other — but every
caller has to consult both, and both handle a reader whose optional dependency
is missing the same way, so they live together.

A reader with a missing dependency never imports and so never registers. That
would leave its formats silently absent from the supported list, which is why
`_OPTIONAL_READERS` restates their suffixes: it is what turns "unsupported
format" into "install `webknossos[czi]`".
"""

from __future__ import annotations

import glob
import os
import warnings
from os import environ

from upath import UPath

from .chunked_image_source import ChunkedImageSource
from .slice_sequence import SliceSequence


class UnknownFormatError(Exception):
    """Raised when no registered reader can open a file."""


_SLICE_READER_CLASSES: list[type[SliceSequence]] = []
_CHUNKED_IMAGE_SOURCE_CLASSES: list[type[ChunkedImageSource]] = []


def register_slice_reader(cls: type[SliceSequence]) -> type[SliceSequence]:
    _SLICE_READER_CLASSES.append(cls)
    return cls


def register_chunked_image_source(
    cls: type[ChunkedImageSource],
) -> type[ChunkedImageSource]:
    _CHUNKED_IMAGE_SOURCE_CLASSES.append(cls)
    return cls


def get_valid_slice_reader_suffixes() -> set[str]:
    """The suffixes (without dot) that some registered slice reader can open."""
    valid_suffixes: set[str] = set()
    for cls in _SLICE_READER_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


def get_valid_chunked_image_suffixes() -> set[str]:
    """The suffixes (without dot) that a registered chunk-based format reads."""
    valid_suffixes: set[str] = set()
    for cls in _CHUNKED_IMAGE_SOURCE_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


def get_valid_suffixes() -> set[str]:
    """Every suffix that can be converted, by either strategy."""
    return get_valid_slice_reader_suffixes() | get_valid_chunked_image_suffixes()


def open_images(path_spec: str, **kwargs: object) -> SliceSequence:
    """Opens a path, glob pattern or directory as a sequence of slices.

    A pattern matching more than one file becomes an image sequence; a single
    file is handed to the registered reader with the highest `class_priority`
    that claims its suffix, falling back to the next one if that reader raises.
    """
    # Deferred to avoid a cycle: the raster readers register themselves here.
    from .raster_slices import MultiImageSlices

    if len(glob.glob(path_spec)) > 1:
        return MultiImageSlices(path_spec, **kwargs)

    _, ext = os.path.splitext(path_spec)
    if len(ext) < 2:
        raise UnknownFormatError(
            f"Could not detect the file type of {path_spec} because it has no "
            "extension."
        )
    ext = ext.lower()[1:]

    eligible_handlers = [
        handler
        for handler in _SLICE_READER_CLASSES
        if ext in {e.lstrip(".").lower() for e in handler.class_exts()}
    ]
    if not eligible_handlers:
        raise UnknownFormatError(
            f"Could not autodetect how to load a file of type {ext}. The "
            f"following suffixes are supported: {sorted(get_valid_suffixes())}"
        )

    messages = []
    for handler in sorted(
        eligible_handlers, key=lambda cls: cls.class_priority, reverse=True
    ):
        try:
            return handler(path_spec, **kwargs)  # type: ignore[call-arg]
        except Exception as e:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
            messages.append(f"{handler.__name__} errored: {e}")
    raise UnknownFormatError(
        "All handlers returned exceptions:\n" + "\n".join(messages)
    )


def try_open_chunked_image_source(
    images: UPath | list[UPath],
    *,
    channel: int | None,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    is_segmentation: bool,
    **format_kwargs: int | None,
) -> ChunkedImageSource | None:
    """
    Returns a ChunkedImageSource if `images` is a single path whose suffix a
    registered chunk-based format handles, else None — leaving the caller to
    fall back to the other strategy. Chunk-based formats are inherently
    single-file, so a list of paths is never one.

    `format_kwargs` may carry knobs only some formats understand (e.g.
    czi_channel); each is passed on only to a class that declares it in
    class_open_kwargs, so callers can pass all of them unconditionally.
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
    for cls in _CHUNKED_IMAGE_SOURCE_CLASSES:
        if suffix in cls.class_exts():
            return cls(
                path,
                **{
                    name: value
                    for name, value in format_kwargs.items()
                    if name in cls.class_open_kwargs
                },
                channel=channel,
                swap_xy=swap_xy,
                flip_x=flip_x,
                flip_y=flip_y,
                flip_z=flip_z,
                is_segmentation=is_segmentation,
            )
    return None


# The extra and suffixes each optional reader is responsible for, for both
# strategies. A reader whose dependency is missing never imports, so it never
# registers and cannot report its own class_exts() — these have to be declared
# out here for the "you are missing an optional dependency" hint to exist at
# all. test_optional_reader_suffixes_match_class_exts keeps them in sync.
_OPTIONAL_READERS: dict[str, tuple[str, frozenset[str]]] = {
    "TiffSlices": ("tifffile", frozenset({"tif", "tiff"})),
    "ImsImageSource": ("ims", frozenset({"ims"})),
    "MrcImageSource": ("mrcfile", frozenset({"mrc", "rec", "st", "map", "ali"})),
    "CziImageSource": ("czi", frozenset({"czi"})),
}

# suffix -> extra, for readers that failed to import. Populated at import time
# below and consumed by get_unavailable_suffixes().
_UNAVAILABLE_SUFFIXES: dict[str, str] = {}


def get_unavailable_suffixes() -> dict[str, str]:
    """
    Maps each suffix that a reader *would* handle, but cannot because its
    optional dependency is missing, to the extra that provides it (e.g.
    {"ims": "ims"}). Empty when every reader imported successfully.

    Lets callers turn "no supported image data found" into a message that
    names the missing dependency, rather than silently omitting the format
    from the supported list.
    """
    return dict(_UNAVAILABLE_SUFFIXES)


def describe_missing_extras(found: dict[str, str]) -> str:
    """
    Turns a suffix -> extra mapping, as returned for the files at hand by
    get_unavailable_suffixes(), into a sentence naming the extras to install.
    Meant to be appended to an error message.
    """
    extras = sorted(set(found.values()))
    return (
        f". Found {', '.join('.' + s for s in sorted(found))} files, which need an "
        + "optional dependency that is not installed — install it with "
        + f"`pip install {' '.join(f'webknossos[{extra}]' for extra in extras)}`"
    )


def _import_readers() -> str | None:
    """Imports every reader module so each one registers itself, and records
    which could not be imported. Returns a description of those, or None."""
    import_exceptions = []

    # No optional dependency beyond imageio, which is a hard requirement — but
    # imported here so that every reader registers in one place.
    from . import (
        dm_slices,  # noqa: F401 unused-import
        raster_slices,  # noqa: F401 unused-import
    )

    for module_name, class_name in (
        ("tiff_slices", "TiffSlices"),
        ("ims_image_source", "ImsImageSource"),
        ("mrc_image_source", "MrcImageSource"),
        ("czi_image_source", "CziImageSource"),
    ):
        try:
            __import__(f"{__package__}.{module_name}", fromlist=[class_name])
        except ImportError as import_error:  # noqa: PERF203
            import_exceptions.append(f"{class_name}: {import_error.msg}")

    registered = {cls.__name__ for cls in _SLICE_READER_CLASSES} | {
        cls.__name__ for cls in _CHUNKED_IMAGE_SOURCE_CLASSES
    }
    for name, (extra, suffixes) in _OPTIONAL_READERS.items():
        if name not in registered:
            for suffix in suffixes:
                _UNAVAILABLE_SUFFIXES[suffix] = extra

    if import_exceptions:
        return "".join(
            f"\t- {import_exception}\n" for import_exception in import_exceptions
        )
    return None


if (_reader_warnings := _import_readers()) is not None:
    if environ.get("WEBKNOSSOS_SHOWED_IMAGE_READER_IMPORT_WARNING", "False") == "False":
        # If the environment variable is not set, we assume that the user has not seen the warning yet.
        # We set it to True to prevent showing the warning again.
        environ["WEBKNOSSOS_SHOWED_IMAGE_READER_IMPORT_WARNING"] = "True"
        warnings.warn(
            f"[WARNING] Not all image readers could be imported:\n{_reader_warnings}Install the readers you need or use 'webknossos[all]' to install all readers.",
            category=UserWarning,
            source=None,
            stacklevel=2,
        )

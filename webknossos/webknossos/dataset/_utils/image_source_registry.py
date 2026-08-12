"""Which reader handles which file suffix, and the one way a source is opened.

`open_image_source()` is the single entry point: it picks the reading strategy
from the suffix and hands back an `ImageSource`, so callers never have to know
which of the two exists. Two kinds of class register here for it to choose
between:

* slice readers (`SliceSequence` subclasses), which `SlicedImageSource` drives
  slice by slice;
* `ChunkedImageSource` subclasses, which read shard-sized blocks themselves.

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
from typing import NamedTuple

from upath import UPath

from .chunked_image_source import ChunkedImageSource
from .image_source import ImageSource, ReadOptions
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


def open_image_source(images: UPath | list[UPath], options: ReadOptions) -> ImageSource:
    """Opens `images` as an `ImageSource`, choosing the reading strategy itself.

    A single file whose suffix a chunk-based format claims is read that way.
    Everything else is read slice by slice — including every list of paths,
    since chunk-based formats are inherently single-file, and callers normalize
    to UPath before getting here (see `_normalize_images_argument`).

    This is the only place the two strategies are told apart, so no caller has
    to ask which kind of source it is holding.
    """
    # Deferred to avoid a cycle: SlicedImageSource consults this registry.
    from .sliced_image_source import SlicedImageSource

    if isinstance(images, UPath):
        suffix = images.suffix.lstrip(".").lower()
        for cls in _CHUNKED_IMAGE_SOURCE_CLASSES:
            if suffix in cls.class_exts():
                # Remote UPaths deliberately do reach the reader, which raises
                # its own "must be a local file path" error via is_remote_path.
                return cls(images, options)
    return SlicedImageSource(images, options)


class _OptionalReader(NamedTuple):
    """One reader that only exists when its extra is installed."""

    module: str
    """The module under this package that defines it."""

    class_name: str
    """Only used to name it in the import warning."""

    extra: str
    """The `webknossos[...]` extra that provides its dependency."""

    suffixes: frozenset[str]
    """What its class_exts() returns. Restated here because a reader whose
    dependency is missing never imports, and so cannot be asked — which is
    exactly when this is needed, to turn "unsupported format" into "install
    `webknossos[czi]`". test_optional_reader_suffixes_match_class_exts keeps
    the two in sync."""


# Every optional reader, declared once. Covers both strategies: slice readers
# are just as optional as chunked ones now that tifffile is not in the base
# install.
_OPTIONAL_READERS: tuple[_OptionalReader, ...] = (
    _OptionalReader(
        "tiff_slices", "TiffSlices", "tifffile", frozenset({"tif", "tiff"})
    ),
    _OptionalReader("ims_image_source", "ImsImageSource", "ims", frozenset({"ims"})),
    _OptionalReader(
        "mrc_image_source",
        "MrcImageSource",
        "mrcfile",
        frozenset({"mrc", "rec", "st", "map", "ali"}),
    ),
    _OptionalReader("czi_image_source", "CziImageSource", "czi", frozenset({"czi"})),
)

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

    for reader in _OPTIONAL_READERS:
        try:
            __import__(f"{__package__}.{reader.module}", fromlist=[reader.class_name])
        except ImportError as import_error:  # noqa: PERF203
            import_exceptions.append(f"{reader.class_name}: {import_error.msg}")
            # The reader never registered, so its formats would silently drop
            # out of the supported list; this is what names the extra instead.
            for suffix in reader.suffixes:
                _UNAVAILABLE_SUFFIXES[suffix] = reader.extra

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

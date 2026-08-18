"""Which reader handles which file extension, and the one way a source is opened.

A **reader** knows one input format and is what registers here; a **source**
is an `ImageSource`. The two kinds of reader differ in whether they are also
sources: a slice reader (`SliceReader` subclass) is not and gets wrapped,
while a chunked reader is, which is what its `…ImageSource` name says.

`open_image_source()` picks the kind from the extension and returns an
`ImageSource` either way, so no caller has to know which it got.
"""

from __future__ import annotations

import glob
import os
import warnings
from os import environ
from typing import NamedTuple

from upath import UPath

from ..errors import UnsupportedImageFormatError
from .chunked_image_source import ChunkedImageSource
from .image_source import ImageSource, ReadOptions
from .slice_reader import SliceReader

_SLICE_READER_CLASSES: list[type[SliceReader]] = []
_CHUNKED_IMAGE_SOURCE_CLASSES: list[type[ChunkedImageSource]] = []


def register_slice_reader(cls: type[SliceReader]) -> type[SliceReader]:
    _SLICE_READER_CLASSES.append(cls)
    return cls


def register_chunked_image_source(
    cls: type[ChunkedImageSource],
) -> type[ChunkedImageSource]:
    _CHUNKED_IMAGE_SOURCE_CLASSES.append(cls)
    return cls


def get_valid_slice_reader_extensions() -> set[str]:
    """The extensions (without dot) that some registered slice reader can open."""
    valid_extensions: set[str] = set()
    for cls in _SLICE_READER_CLASSES:
        valid_extensions.update(cls.supported_file_extensions())
    return valid_extensions


def get_valid_chunked_reader_extensions() -> set[str]:
    """The extensions (without dot) that some registered chunked reader can open."""
    valid_extensions: set[str] = set()
    for cls in _CHUNKED_IMAGE_SOURCE_CLASSES:
        valid_extensions.update(cls.supported_file_extensions())
    return valid_extensions


def get_valid_extensions() -> set[str]:
    """Every extension that can be converted, by a reader of either kind."""
    return get_valid_slice_reader_extensions() | get_valid_chunked_reader_extensions()


def open_slice_reader(path_spec: str, **kwargs: object) -> SliceReader:
    """Opens a path, glob pattern or directory as a sequence of slices.

    A pattern matching more than one file becomes an image sequence; a single
    file is handed to the registered reader with the highest `class_priority`
    that claims its extension, falling back to the next one if that reader
    raises.
    """
    # Deferred to avoid a cycle: the raster readers register themselves here.
    from .common_slice_readers import MultiImageSliceReader

    if len(glob.glob(path_spec)) > 1:
        return MultiImageSliceReader(path_spec, **kwargs)

    # The errors below are dispatch failing, not the final word: a caller may
    # replace them with one that knows the actual path and missing extras
    # (path_spec may be a glob, so these cannot).
    supported = tuple(sorted(get_valid_extensions()))

    _, ext = os.path.splitext(path_spec)
    if len(ext) < 2:
        raise UnsupportedImageFormatError(
            f"Could not detect the file type of {path_spec} because it has no "
            "extension.",
            supported_file_extensions=supported,
        )
    ext = ext.lower()[1:]

    eligible_handlers = [
        handler
        for handler in _SLICE_READER_CLASSES
        if ext in {e.lstrip(".").lower() for e in handler.supported_file_extensions()}
    ]
    if not eligible_handlers:
        raise UnsupportedImageFormatError(
            f"Could not autodetect how to load a file of type {ext}. The "
            f"following extensions are supported: {list(supported)}",
            file_extension=ext,
            supported_file_extensions=supported,
        )

    messages = []
    for handler in sorted(
        eligible_handlers, key=lambda cls: cls.class_priority, reverse=True
    ):
        try:
            return handler(path_spec, **kwargs)  # type: ignore[call-arg]
        except Exception as e:  # noqa: PERF203 `try`-`except` within a loop incurs performance overhead
            messages.append(f"{handler.__name__} errored: {e}")
    raise UnsupportedImageFormatError(
        "All handlers returned exceptions:\n" + "\n".join(messages),
        file_extension=ext,
        supported_file_extensions=supported,
    )


def open_image_source(images: UPath | list[UPath], options: ReadOptions) -> ImageSource:
    """Opens `images` as an `ImageSource`, choosing the reader kind itself.

    A single file whose extension a chunked reader claims is read that way;
    everything else slice by slice, including every list of paths, since
    chunked formats are inherently single-file.
    """
    # Deferred to avoid a cycle: SlicedImageSource consults this registry.
    from .sliced_image_source import SlicedImageSource

    if isinstance(images, UPath):
        extension = images.suffix.lstrip(".").lower()
        for cls in _CHUNKED_IMAGE_SOURCE_CLASSES:
            if extension in cls.supported_file_extensions():
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

    extensions: frozenset[str]
    """The extensions this reader supports, restated here because a reader
    whose dependency is missing never imports and so cannot be asked."""


_OPTIONAL_SLICE_READERS_AND_IMAGE_SOURCES: tuple[_OptionalReader, ...] = (
    _OptionalReader(
        "tiff_slice_reader", "TiffSliceReader", "tifffile", frozenset({"tif", "tiff"})
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

# extension -> extra, for readers that failed to import. Populated at import
# time below and consumed by get_unavailable_extensions().
_UNAVAILABLE_EXTENSIONS: dict[str, str] = {}


def get_unavailable_extensions() -> dict[str, str]:
    """
    Maps each extension a reader *would* handle, but cannot because its
    optional dependency is missing, to the extra that provides it (e.g.
    {"ims": "ims"}). Empty when every reader imported. Lets callers name the
    missing dependency instead of silently omitting the format from the
    supported list.
    """
    return dict(_UNAVAILABLE_EXTENSIONS)


def describe_missing_extras(found: dict[str, str]) -> str:
    """
    Turns a get_unavailable_extensions() mapping, narrowed to the files at
    hand, into a sentence naming the extras to install. Appended to an error
    message.
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
        common_slice_readers,  # noqa: F401 unused-import
        dm_slice_reader,  # noqa: F401 unused-import
    )

    for reader in _OPTIONAL_SLICE_READERS_AND_IMAGE_SOURCES:
        try:
            __import__(f"{__package__}.{reader.module}", fromlist=[reader.class_name])
        except ImportError as import_error:  # noqa: PERF203
            import_exceptions.append(f"{reader.class_name}: {import_error.msg}")
            # The reader never registered, so its formats would silently drop
            # out of the supported list; this is what names the extra instead.
            for extension in reader.extensions:
                _UNAVAILABLE_EXTENSIONS[extension] = reader.extra

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

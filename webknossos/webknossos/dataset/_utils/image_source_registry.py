"""Which reader handles which file suffix, and the one way a source is opened.

A **reader** knows one input format and is what registers here; a **source** is
an `ImageSource`, what `image_conversion.py` drives. The two kinds of reader
differ in whether they are also sources: a slice reader (`SliceSequence`
subclass) is not — `SlicedImageSource` wraps it — while a chunked reader is,
which is what its `…ImageSource` name says.

`open_image_source()` picks the kind from the suffix and returns an
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
from .slice_sequence import SliceSequence

_SLICE_READER_CLASSES: list[type[SliceSequence]] = []
_CHUNKED_READER_CLASSES: list[type[ChunkedImageSource]] = []


def register_slice_reader(cls: type[SliceSequence]) -> type[SliceSequence]:
    _SLICE_READER_CLASSES.append(cls)
    return cls


def register_chunked_reader(
    cls: type[ChunkedImageSource],
) -> type[ChunkedImageSource]:
    _CHUNKED_READER_CLASSES.append(cls)
    return cls


def get_valid_slice_reader_suffixes() -> set[str]:
    """The suffixes (without dot) that some registered slice reader can open."""
    valid_suffixes: set[str] = set()
    for cls in _SLICE_READER_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


def get_valid_chunked_reader_suffixes() -> set[str]:
    """The suffixes (without dot) that some registered chunked reader can open."""
    valid_suffixes: set[str] = set()
    for cls in _CHUNKED_READER_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


def get_valid_suffixes() -> set[str]:
    """Every suffix that can be converted, by a reader of either kind."""
    return get_valid_slice_reader_suffixes() | get_valid_chunked_reader_suffixes()


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

    # The errors below are dispatch failing, not the final word: they are
    # collected while SlicedImageSource tries its other open strategies, and
    # _classify_open_failure replaces them with one that knows the `path` and
    # `missing_extras` (path_spec may be a glob, so these cannot).
    supported = tuple(sorted(get_valid_suffixes()))

    _, ext = os.path.splitext(path_spec)
    if len(ext) < 2:
        raise UnsupportedImageFormatError(
            f"Could not detect the file type of {path_spec} because it has no "
            "extension.",
            supported_suffixes=supported,
        )
    ext = ext.lower()[1:]

    eligible_handlers = [
        handler
        for handler in _SLICE_READER_CLASSES
        if ext in {e.lstrip(".").lower() for e in handler.class_exts()}
    ]
    if not eligible_handlers:
        raise UnsupportedImageFormatError(
            f"Could not autodetect how to load a file of type {ext}. The "
            f"following suffixes are supported: {list(supported)}",
            suffix=ext,
            supported_suffixes=supported,
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
        suffix=ext,
        supported_suffixes=supported,
    )


def _chunked_reader_for_directory(path: UPath) -> type[ChunkedImageSource] | None:
    """The registered chunked reader that claims `path` as a directory-based
    store, by suffix first and then by content (`probe_directory`). None if
    no reader claims it — e.g. an ordinary directory of images."""
    suffix = path.suffix.lstrip(".").lower()
    for cls in _CHUNKED_READER_CLASSES:
        if suffix in cls.class_exts():
            return cls
    for cls in _CHUNKED_READER_CLASSES:
        if cls.probe_directory(path):
            return cls
    return None


def is_chunked_source_directory(path: UPath) -> bool:
    """Whether `path` is a directory that some registered chunked reader would
    open as a single store (a Zarr/N5/neuroglancer-precomputed root), rather
    than a directory to walk into for individual image files. Used by the
    directory walk in image_conversion.py so it treats such a directory as one
    leaf item, the same way it already treats a single `.ims`/`.mrc` file."""
    return path.is_dir() and _chunked_reader_for_directory(path) is not None


def open_image_source(images: UPath | list[UPath], options: ReadOptions) -> ImageSource:
    """Opens `images` as an `ImageSource`, choosing the reader kind itself.

    A single file whose suffix a chunked reader claims is read that way. A
    directory is also read that way if some chunked reader recognizes it by
    suffix or by content (`probe_directory`) — e.g. a Zarr/N5 store, whether
    or not it is named `*.zarr`/`*.n5`, or a neuroglancer precomputed volume,
    which has no suffix convention at all. Everything else is read slice by
    slice, including every list of paths, since chunked formats are inherently
    single-file (or single-directory).
    """
    # Deferred to avoid a cycle: SlicedImageSource consults this registry.
    from .sliced_image_source import SlicedImageSource

    if isinstance(images, UPath):
        suffix = images.suffix.lstrip(".").lower()
        for cls in _CHUNKED_READER_CLASSES:
            if suffix in cls.class_exts():
                # Remote UPaths deliberately do reach the reader, which raises
                # its own "must be a local file path" error via is_remote_path.
                return cls(images, options)
        if images.is_dir():
            dir_cls = _chunked_reader_for_directory(images)
            if dir_cls is not None:
                return dir_cls(images, options)
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
    """What its class_exts() returns, restated because a reader whose
    dependency is missing never imports and so cannot be asked — which is
    exactly when this is needed. test_optional_reader_suffixes_match_class_exts
    keeps the two in sync."""


# Every optional reader, declared once. Slice readers are as optional as
# chunked ones now that tifffile is not in the base install.
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
    Maps each suffix a reader *would* handle, but cannot because its optional
    dependency is missing, to the extra that provides it (e.g. {"ims": "ims"}).
    Empty when every reader imported. Lets callers name the missing dependency
    instead of silently omitting the format from the supported list.
    """
    return dict(_UNAVAILABLE_SUFFIXES)


def describe_missing_extras(found: dict[str, str]) -> str:
    """
    Turns a get_unavailable_suffixes() mapping, narrowed to the files at hand,
    into a sentence naming the extras to install. Appended to an error message.
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

    # No optional dependency beyond imageio/tensorstore, which are hard
    # requirements — but imported here so that every reader registers in one
    # place. tensorstore already reads zarr/zarr3/n5/neuroglancer_precomputed
    # for webknossos's own dataset storage, so these need no extra.
    from . import (
        dm_slices,  # noqa: F401 unused-import
        n5_image_source,  # noqa: F401 unused-import
        neuroglancer_precomputed_image_source,  # noqa: F401 unused-import
        raster_slices,  # noqa: F401 unused-import
        zarr_image_source,  # noqa: F401 unused-import
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

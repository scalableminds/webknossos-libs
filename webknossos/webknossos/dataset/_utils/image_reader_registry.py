"""Registry mapping file suffixes to slice-by-slice image readers.

Replaces `pims.open()`. Readers opt in with `@register_image_reader` instead of
being discovered by walking subclasses, mirroring the `register_chunked_images`
pattern in `chunked_images.py`.
"""

from __future__ import annotations

import glob
import os
import warnings
from os import environ

from .frame_sequence import FrameSequence


class UnknownFormatError(Exception):
    """Raised when no registered reader can open a file."""


_IMAGE_READER_CLASSES: list[type[FrameSequence]] = []


def register_image_reader(cls: type[FrameSequence]) -> type[FrameSequence]:
    _IMAGE_READER_CLASSES.append(cls)
    return cls


def get_valid_image_suffixes() -> set[str]:
    """The suffixes (without dot) that some registered reader can open."""
    valid_suffixes: set[str] = set()
    for cls in _IMAGE_READER_CLASSES:
        valid_suffixes.update(cls.class_exts())
    return valid_suffixes


def open_images(path_spec: str, **kwargs: object) -> FrameSequence:
    """Opens a path, glob pattern or directory as a sequence of frames.

    A pattern matching more than one file becomes an image sequence; a single
    file is handed to the registered reader with the highest `class_priority`
    that claims its suffix, falling back to the next one if that reader raises.
    """
    # Deferred to avoid a cycle: the raster readers register themselves here.
    from .raster_image_readers import ImageSequenceReader

    if len(glob.glob(path_spec)) > 1:
        return ImageSequenceReader(path_spec, **kwargs)

    _, ext = os.path.splitext(path_spec)
    if len(ext) < 2:
        raise UnknownFormatError(
            f"Could not detect the file type of {path_spec} because it has no "
            "extension."
        )
    ext = ext.lower()[1:]

    eligible_handlers = [
        handler
        for handler in _IMAGE_READER_CLASSES
        if ext in {e.lstrip(".").lower() for e in handler.class_exts()}
    ]
    if not eligible_handlers:
        raise UnknownFormatError(
            f"Could not autodetect how to load a file of type {ext}. The "
            f"following suffixes are supported: {sorted(get_valid_image_suffixes())}"
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


def _image_reader_imports() -> str | None:
    import_exceptions = []

    # No optional dependency of its own beyond imageio, which is a hard
    # requirement — but keep it here so every reader registers in one place.
    from . import (
        dm_sequence_readers,  # noqa: F401 unused-import
        raster_image_readers,  # noqa: F401 unused-import
    )

    try:
        from .czi_sequence_reader import CziSequenceReader  # noqa: F401 unused-import
    except ImportError as import_error:
        import_exceptions.append(f"CziSequenceReader: {import_error.msg}")

    try:
        from .tiff_sequence_reader import TiffSequenceReader  # noqa: F401 unused-import
    except ImportError as import_error:
        import_exceptions.append(f"TiffSequenceReader: {import_error.msg}")

    if import_exceptions:
        return "".join(
            f"\t- {import_exception}\n" for import_exception in import_exceptions
        )
    return None


if (image_reader_warnings := _image_reader_imports()) is not None:
    if environ.get("WEBKNOSSOS_SHOWED_IMAGE_READER_IMPORT_WARNING", "False") == "False":
        # If the environment variable is not set, we assume that the user has not seen the warning yet.
        # We set it to True to prevent showing the warning again.
        environ["WEBKNOSSOS_SHOWED_IMAGE_READER_IMPORT_WARNING"] = "True"
        warnings.warn(
            f"[WARNING] Not all image readers could be imported:\n{image_reader_warnings}Install the readers you need or use 'webknossos[all]' to install all readers.",
            category=UserWarning,
            source=None,
            stacklevel=2,
        )

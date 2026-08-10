"""Exceptions raised by the dataset API.

This module deliberately imports nothing but `upath`, so that downstream code
can catch these exceptions without pulling in the image readers or any of
their optional dependencies.
"""

from __future__ import annotations

from upath import UPath


class ImageConversionError(ValueError):
    """Base class for the ways image conversion can fail because of its input.

    Raised by `Dataset.from_images` and `Dataset.add_layer_from_images`. Catch
    this to handle every such failure at once, or one of its subclasses to tell
    them apart — they map to different instructions for the user:

    * `UnsupportedImageFormatError`: no reader handles this format at all.
    * `CorruptImageError`: the format is supported, but the file could not be
      read — typically damaged or incompletely uploaded.
    * `UnsupportedImageDataError`: the file was read fine, but its data cannot
      be stored as requested.

    Subclasses `ValueError`, which the call sites raised before these exceptions
    existed, so `except ValueError` keeps working.

    Attributes:
        path: The offending input path, if there is a single one. `None` when
            the images were passed as a list or as a `pims.FramesSequence`, or
            when the failure is not tied to one file.
    """

    def __init__(self, message: str, *, path: UPath | None = None) -> None:
        super().__init__(message)
        self.path = path


class UnsupportedImageFormatError(ImageConversionError):
    """Raised when image data cannot be converted because no reader handles its format.

    Raised either because the input contains no file with a supported suffix,
    or because every reader failed to recognize the file.

    Attributes:
        path: The offending input path, if there is a single one. `None` when
            the images were passed as a list or as a `pims.FramesSequence`.
        suffix: The offending file's suffix, lowercase and without the leading
            dot (e.g. `"dcm"`). `None` when `path` is a directory or unknown.
        supported_suffixes: The suffixes that can currently be converted, in
            the same lowercase, dot-less form.
        missing_extras: The `webknossos` extras that would add support for the
            input at hand, e.g. `("ims",)` when converting an `.ims` file
            without `webknossos[ims]` installed. Empty when the format is not
            supported at all — which is the distinction to make when turning
            this exception into a user-facing message: a non-empty value means
            the format *is* supported and the installation is incomplete.

    Examples:
        ```
        try:
            wk.Dataset.from_images(input_path, output_path, voxel_size=(1, 1, 1))
        except wk.UnsupportedImageFormatError as e:
            if e.missing_extras:
                print(f"Install webknossos[{','.join(e.missing_extras)}] to convert this file.")
            else:
                print(f"Cannot convert .{e.suffix} files.")
        ```
    """

    def __init__(
        self,
        message: str,
        *,
        path: UPath | None = None,
        suffix: str | None = None,
        supported_suffixes: tuple[str, ...] = (),
        missing_extras: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message, path=path)
        self.suffix = suffix
        self.supported_suffixes = supported_suffixes
        self.missing_extras = missing_extras


class CorruptImageError(ImageConversionError):
    """Raised when a file of a supported format cannot be read.

    The reader recognized the format but could not make sense of the contents,
    which in practice almost always means the file is damaged or was uploaded
    incompletely. The underlying reader error is kept as the `__cause__`, which
    belongs in a log rather than in a user-facing message.

    Missing files and permission errors are deliberately *not* reported this
    way: those keep raising `FileNotFoundError`/`PermissionError`.
    """


class UnsupportedImageDataError(ImageConversionError):
    """Raised when readable image data cannot be stored as requested.

    The file itself is fine — its data does not fit the target, e.g. a
    segmentation layer with a non-integer dtype, an image with more axes than
    the chosen `data_format` supports, or a dimensionality no reader can map to
    a layer. The message names what to change; unlike a `CorruptImageError`,
    these are usually fixable by converting with different arguments.
    """

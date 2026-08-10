"""Exceptions raised by the dataset API.

This module deliberately imports nothing but `upath`, so that downstream code
can catch these exceptions without pulling in the image readers or any of
their optional dependencies.
"""

from __future__ import annotations

from upath import UPath


class UnsupportedImageFormatError(ValueError):
    """Raised when image data cannot be converted because no reader handles its format.

    Raised by `Dataset.from_images` and `Dataset.add_layer_from_images`, either
    because the input contains no file with a supported suffix, or because every
    reader failed to recognize the file.

    Subclasses `ValueError`, which both call sites raised before this exception
    existed, so `except ValueError` keeps working.

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
        super().__init__(message)
        self.path = path
        self.suffix = suffix
        self.supported_suffixes = supported_suffixes
        self.missing_extras = missing_extras

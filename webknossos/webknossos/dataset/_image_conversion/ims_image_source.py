from __future__ import annotations

import contextlib
import gc
import io

import h5py
import numpy as np
from upath import UPath

from ...utils import WkImportError, is_remote_path
from ..errors import CorruptImageError
from .chunked_image_source import ChunkedImageSource
from .image_source import ReadOptions, compute_channel_selection
from .image_source_registry import register_chunked_image_source

try:
    from imaris_ims_file_reader.ims import ims as ImsFile
    from imaris_ims_file_reader.ims import ims_reader as _ImsReader
except ImportError as e:
    raise WkImportError("imaris-ims-file-reader", "ims") from e

# The reader only assigns `hf` once h5py has opened the file, so a corrupt file
# leaves a half-built object whose __del__ raises AttributeError as an
# unraisable exception. A class-level default makes close() a no-op instead.
if not hasattr(_ImsReader, "hf"):
    _ImsReader.hf = None


def _read_ims_metadata_quietly(
    path: str,
) -> tuple[tuple[int, int, int, int, int], np.dtype]:
    # The published imaris-ims-file-reader (as of 0.1.8) unconditionally prints
    # "Opening readonly file: ..." / "Closing file: ..." and has no `verbose`
    # kwarg to suppress it. It also calls close() again from __del__, so the
    # object must be closed, deleted, and garbage-collected while stdout is
    # still redirected — otherwise that second "Closing file" print leaks out
    # once the object is finalized after this function returns.
    with contextlib.redirect_stdout(io.StringIO()):
        ims_obj = ImsFile(path, squeeze_output=False)
        shape = tuple(int(s) for s in ims_obj.shape)
        dtype = np.dtype(ims_obj.dtype)
        ims_obj.close()
        del ims_obj
        gc.collect()
    return shape, dtype  # type: ignore[return-value]


@register_chunked_image_source
class ImsImageSource(ChunkedImageSource):
    """
    ChunkedImageSource for Imaris .ims files. Reads shard-sized 3D blocks
    straight out of the underlying HDF5 file via h5py and writes them to
    mag_view.
    """

    @classmethod
    def supported_file_extensions(cls) -> set[str]:
        return {"ims"}

    def __init__(self, path: UPath, options: ReadOptions) -> None:
        super().__init__(path, options)
        if is_remote_path(path):
            raise ValueError(
                f"Cannot open IMS file from {path}. The path must be a local file path."
            )

        try:
            file_shape, self.dtype = _read_ims_metadata_quietly(str(path))
        except Exception as e:
            # The reader speaks HDF5, so a damaged or truncated file surfaces
            # as an OSError/KeyError from h5py deep inside it, which says
            # nothing useful to whoever uploaded the file.
            raise CorruptImageError(
                f"Cannot open IMS file {path}. "
                + "The file is likely corrupted or not a valid IMS file.",
                path=path,
            ) from e
        t, raw_num_channels, self._z, self._y, self._x = file_shape
        self._t = t

        if 0 in (t, raw_num_channels, self._z, self._y, self._x):
            # A damaged or incompletely uploaded .ims file can still open and
            # report a shape, just with an empty axis, which would otherwise
            # convert into an empty layer or fail much later with an
            # unrelated error.
            raise CorruptImageError(
                f"Cannot open IMS file {path}. It describes an empty image "
                + f"(shape {file_shape}), so the file is likely corrupted or "
                + "not a valid IMS file.",
                path=path,
            )

        self.num_channels, self._channel, self._first_n_channels, possible_channels = (
            compute_channel_selection(raw_num_channels, options.channel)
        )
        self._possible_layers: dict[str, list[int]] = {}
        if possible_channels is not None:
            self._possible_layers["channel"] = possible_channels

        # A "t" axis is only added to the bounding box when there actually are
        # multiple timepoints; each chunk along that axis then carries its own
        # timepoint (chunk size 1). A single-timepoint file stays 3D and
        # always reads timepoint 0.
        self._include_t_axis = t > 1
        self._fixed_timepoint: int | None = None if self._include_t_axis else 0

    def get_layer_split_options(self) -> dict[str, list[int]] | None:
        if len(self._possible_layers) == 0:
            return None
        return self._possible_layers

    def _read_source_box(
        self,
        *,
        timepoint: int,
        z: slice,
        y: slice,
        x: slice,
    ) -> np.ndarray:
        if self._channel is not None:
            channels_to_read = [self._channel]
        elif self._first_n_channels is not None:
            channels_to_read = list(range(self._first_n_channels))
        else:
            channels_to_read = list(range(self.num_channels))

        with h5py.File(str(self.path), "r") as hf:
            ds = hf["DataSet"]
            slabs = [
                # Read exactly the 3D block needed — one decompression per
                # HDF5 chunk row.
                ds[f"ResolutionLevel 0/TimePoint {timepoint}/Channel {ci}/Data"][
                    z, y, x
                ]
                for ci in channels_to_read
            ]
        return np.stack(slabs, axis=0)  # (c, z, y, x)

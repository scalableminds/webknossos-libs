from enum import Enum

from ..dataset_properties import DataFormat


class RemoteAccessMode(Enum):
    """Determines how data of a remote dataset is accessed. Note that DIRECT_PATH can only be used if the client has access to the underlying storage."""

    ZARR_STREAMING = "zarr_streaming"
    DIRECT_PATH = "direct_path"
    PROXY_PATH = "proxy_path"


def data_format_for_access_mode(
    access_mode: RemoteAccessMode, layer_data_format: DataFormat
) -> DataFormat:
    """The data format that is actually served for the given access mode.

    Zarr streaming re-serves the data as Zarr, regardless of how it is stored.
    The proxy is a byte-level proxy, so it preserves the underlying format.
    """
    if access_mode == RemoteAccessMode.ZARR_STREAMING:
        return DataFormat.Zarr
    return layer_data_format

from enum import Enum

from ..dataset_properties import DataFormat
from ..geometry.mag import Mag


class RemoteAccessMode(Enum):
    """Determines how data of a remote dataset is accessed. Note that DIRECT_PATH can only be used if the client has access to the underlying storage."""

    ZARR_STREAMING = "zarr_streaming"
    DIRECT_PATH = "direct_path"
    PROXY_PATH = "proxy_path"


def mag_url_suffix(access_mode: RemoteAccessMode, layer_name: str, mag: Mag) -> str:
    """The path of a mag relative to the base path of the given access mode.

    These layouts are dictated by the WEBKNOSSOS datastore routes. They are kept here
    so that a server-side change only has to be reflected in one place.
    """
    if access_mode == RemoteAccessMode.ZARR_STREAMING:
        return f"{layer_name}/{mag.to_layer_name()}"
    elif access_mode == RemoteAccessMode.PROXY_PATH:
        return f"layers/{layer_name}/mags/{mag.to_layer_name()}"
    else:
        raise ValueError(f"{access_mode} does not have a computed mag path.")


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

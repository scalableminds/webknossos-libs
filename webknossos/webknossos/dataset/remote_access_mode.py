from enum import Enum


class RemoteAccessMode(Enum):
    """Determines how data of a remote dataset is accessed. Note that DIRECT_PATH can only be used if the client has access to the underlying storage."""

    ZARR_STREAMING = "zarr_streaming"
    DIRECT_PATH = "direct_path"
    PROXY_PATH = "proxy_path"

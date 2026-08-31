from enum import Enum


class RemoteAccessMode(Enum):
    """Determines how data of a remote dataset is accessed.
    
`ZARR_STREAMING`: The WEBKNOSSOS datastore re-serves the data as Zarr. Works everywhere and is the default. It is also the only mode that can read annotations. 
`PROXY_PATH`: The WEBKNOSSOS datastore proxies the bytes of the underlying storage, preserving its data format. 
`DIRECT_PATH`: The underlying storage is read directly. This is the fastest option, but it requires that your machine has access to that storage (and credentials for it). 

    ZARR_STREAMING = "zarr_streaming"
    DIRECT_PATH = "direct_path"
    PROXY_PATH = "proxy_path"

from typing import TYPE_CHECKING

from upath import UPath

from ....dataset_properties import MagViewProperties
from ....geometry import Mag
from ...remote_access_mode import RemoteAccessMode, data_format_for_access_mode
from .mag_view import MagView

if TYPE_CHECKING:
    from ..remote_layer import RemoteLayer


class RemoteMagView(MagView["RemoteLayer"]):
    """A mag of a remote dataset, accessed in one particular `RemoteAccessMode`.

    Each mag can be accessed independently, so different mags of the same layer may use
    different access modes. Only the direct path is stored in the dataset properties;
    the zarr streaming and proxy paths are computed from the datastore url on demand.

    Examples:
        ```
        ds = RemoteDataset.open("https://webknossos.org/datasets/...")
        layer = ds.get_layer("color")
        mag1 = layer.get_mag(1, access_mode=RemoteAccessMode.DIRECT_PATH)
        mag2 = layer.get_mag(2, access_mode=RemoteAccessMode.PROXY_PATH)
        ```
    """

    _access_mode: RemoteAccessMode

    def __init__(
        self,
        layer: "RemoteLayer",
        mag: Mag,
        *,
        access_mode: RemoteAccessMode,
        read_only: bool = True,
    ) -> None:
        """
        Do not use this constructor manually. Instead use `RemoteLayer.get_mag()`.
        """
        self._access_mode = access_mode
        path = layer.dataset._mag_path(
            layer.name, mag, _mag_properties(layer, mag), access_mode
        )
        super().__init__(
            layer,
            mag,
            path,
            read_only=read_only,
            data_format=data_format_for_access_mode(access_mode, layer.data_format),
        )

    @property
    def access_mode(self) -> RemoteAccessMode:
        """How the data of this mag is accessed."""
        return self._access_mode

    @property
    def direct_path(self) -> UPath | None:
        """The path of the underlying storage, or `None` if the server does not expose it.

        Reading from it requires that the client has access to that storage.
        """
        try:
            return self._path_for(RemoteAccessMode.DIRECT_PATH)
        except ValueError:
            return None

    @property
    def proxy_path(self) -> UPath:
        """The path that proxies the underlying storage through the WEBKNOSSOS datastore."""
        return self._path_for(RemoteAccessMode.PROXY_PATH)

    @property
    def zarr_streaming_path(self) -> UPath:
        """The path where the WEBKNOSSOS datastore re-serves this mag as Zarr."""
        return self._path_for(RemoteAccessMode.ZARR_STREAMING)

    def _path_for(self, access_mode: RemoteAccessMode) -> UPath:
        return self.layer.dataset._mag_path(
            self.layer.name, self._mag, self._properties, access_mode
        )

    def __repr__(self) -> str:
        return (
            f"RemoteMagView(name={repr(self.name)}, access_mode={self._access_mode.value}, "
            + f"bounding_box={self.bounding_box})"
        )


def _mag_properties(layer: "RemoteLayer", mag: Mag) -> MagViewProperties:
    return next(
        mag_properties
        for mag_properties in layer._properties.mags
        if Mag(mag_properties.mag) == mag
    )

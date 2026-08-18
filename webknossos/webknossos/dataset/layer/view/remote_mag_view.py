from collections.abc import Mapping
from types import MappingProxyType
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

    def with_access_mode(self, access_mode: RemoteAccessMode) -> "RemoteMagView":
        """Returns a view of this mag that resolves its path via `access_mode`.

        Raises:
            ValueError: If `access_mode` is not available for this mag.
        """
        return RemoteMagView(
            self.layer, self._mag, access_mode=access_mode, read_only=self._read_only
        )

    @property
    def paths(self) -> Mapping[RemoteAccessMode, UPath]:
        """All paths at which this mag's data can be reached, keyed by access mode.

        Comparing these lets a caller pick which access mode to use for this mag, e.g.
        preferring `DIRECT_PATH` when reachable and falling back otherwise, without
        constructing a separate `RemoteMagView` per candidate mode. A mode is omitted
        if it is not available for this mag (e.g. `DIRECT_PATH` when the server does not
        expose it, or `PROXY_PATH`/`ZARR_STREAMING` for an annotation's volume layers).
        """
        result: dict[RemoteAccessMode, UPath] = {}
        for access_mode in RemoteAccessMode:
            try:
                result[access_mode] = self._path_for(access_mode)
            except ValueError:  # noqa: PERF203 only 3 iterations, clarity wins here
                continue
        return MappingProxyType(result)

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

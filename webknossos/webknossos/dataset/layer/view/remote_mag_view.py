from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from upath import UPath

from ....geometry import Mag, NDBoundingBox, NormalizedBoundingBox
from ....utils import enrich_path
from ...remote_access_mode import RemoteAccessMode
from .mag_view import MagView

if TYPE_CHECKING:
    from ....dataset_properties import LayerProperties
    from ..remote_layer import RemoteLayer


class RemoteMagView(MagView["RemoteLayer"]):
    """A mag of a remote dataset, accessed in one particular `RemoteAccessMode`.

    Each mag can be accessed independently, so different mags of the same layer may use
    different access modes. Only the direct path is stored in the dataset properties;
    the zarr streaming and proxy paths are only known once fetched from their own
    endpoint (see `RemoteDataset._get_dataset_properties_for_mode`).

    Examples:
        ```
        ds = RemoteDataset.open("https://webknossos.org/datasets/...")
        layer = ds.get_layer("color")
        mag1 = layer.get_mag(1, access_mode=RemoteAccessMode.DIRECT_PATH)
        mag2 = layer.get_mag(2, access_mode=RemoteAccessMode.PROXY_PATH)
        ```
    """

    _access_mode: RemoteAccessMode
    _layer_properties: "LayerProperties"

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
        # The properties document actually served by access_mode is the only source
        # that reliably describes what it serves: the datastore may derive a
        # materially different representation for one mode than another mode's
        # document describes (e.g. splitting a multi-channel source into
        # single-channel layers), so bounding_box, data_format and the mag path all
        # come from this single document, never mixed across modes.
        self._layer_properties = layer.dataset._get_layer_properties_for_mode(
            layer.name, access_mode
        )
        mag_properties = next(
            (m for m in self._layer_properties.mags if Mag(m.mag) == mag), None
        )
        if mag_properties is None or mag_properties.path is None:
            raise ValueError(
                f"Cannot access {layer.name}/{mag.to_layer_name()} via "
                + f"{access_mode.value}: not available for this dataset."
            )
        path = enrich_path(mag_properties.path, layer.dataset._base_path(access_mode))
        super().__init__(
            layer,
            mag,
            path,
            read_only=read_only,
            data_format=self._layer_properties.data_format,
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
    def bounding_box(self) -> NDBoundingBox:
        # Overrides MagView's method, which uses self.layer.bounding_box -- the
        # dataset's default access mode's bounding box, which does not necessarily
        # match this mag's own access_mode (see __init__).
        return self.normalized_bounding_box.denormalize()

    @property
    def normalized_bounding_box(self) -> NormalizedBoundingBox:
        return self._layer_properties.bounding_box.align_with_mag(self._mag, ceil=True)

    @property
    def num_channels(self) -> int:
        return self.normalized_bounding_box.size.get("c", 1)

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
                result[access_mode] = self.layer.dataset._mag_path(
                    self.layer.name, self._mag, access_mode
                )
            except ValueError:  # noqa: PERF203 only 3 iterations, clarity wins here
                continue
        return MappingProxyType(result)

    def __repr__(self) -> str:
        return (
            f"RemoteMagView(name={repr(self.name)}, access_mode={self._access_mode.value}, "
            + f"bounding_box={self.bounding_box})"
        )

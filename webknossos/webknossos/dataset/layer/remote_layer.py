from os import PathLike
from typing import TYPE_CHECKING

from upath import UPath

from webknossos.dataset_properties import LayerProperties, MagViewProperties

from ...client.api_client.models import ApiReserveMagUploadToPathParameters
from ...geometry.mag import Mag, MagLike
from ...utils import enrich_path
from .abstract_layer import AbstractLayer
from .view import MagView

if TYPE_CHECKING:
    from webknossos.dataset import RemoteDataset

    from .segmentation_layer import RemoteSegmentationLayer


class RemoteLayer(AbstractLayer):
    _dataset: "RemoteDataset"
    _mags: dict[Mag, MagView["RemoteLayer"]]

    def __init__(
        self, dataset: "RemoteDataset", properties: LayerProperties, read_only: bool
    ):
        super().__init__(dataset, properties, read_only)

    def _determine_read_only_and_path_for_mag(
        self, mag_properties: MagViewProperties
    ) -> tuple[bool, UPath]:
        assert mag_properties.path is not None, (
            f"Remote mags must have a path: {mag_properties}"
        )
        # In case of zarr-streaming remote datasets, the mag paths are relative to the dataset path.
        # In the case of non-streaming remote datasets, the mag paths are absolute.
        mag_path = enrich_path(mag_properties.path, self._dataset.zarr_streaming_path)
        read_only = True
        return read_only, mag_path

    @property
    def dataset(self) -> "RemoteDataset":
        return self._dataset

    def as_segmentation_layer(self) -> "RemoteSegmentationLayer":
        """Casts into SegmentationLayer."""
        from .segmentation_layer import RemoteSegmentationLayer

        if isinstance(self, RemoteSegmentationLayer):
            return self
        else:
            raise TypeError(f"self is not a SegmentationLayer. Got: {type(self)}")

    def get_mag(self, mag: MagLike) -> MagView["RemoteLayer"]:
        return super().get_mag(mag)

    def get_finest_mag(self) -> MagView["RemoteLayer"]:
        return super().get_finest_mag()

    def add_mag_as_copy(
        self, foreign_mag_view_or_path: PathLike | UPath | str | MagView
    ):
        self._ensure_writable()
        foreign_mag_view = MagView._ensure_mag_view(foreign_mag_view_or_path)

        from ...client.context import _get_api_client

        with self._dataset._context:
            client = _get_api_client()
            reserve_parameters = ApiReserveMagUploadToPathParameters(
                layer_name=self.name,
                mag=foreign_mag_view.mag.to_list(),
                axis_order=None,
                channel_index=None,
                path_prefix=None,
                overwrite_pending=True,
            )
            path = client.reserve_mag_upload_to_paths(
                self._dataset.dataset_id, reserve_parameters
            )
            print(f"writing new mag to {path}...")
            # TODO write actual data
            client.finish_mag_upload_to_paths(
                self._dataset.dataset_id, reserve_parameters
            )
        self._apply_server_layer_properties()

    def delete_mag(self, mag: MagLike) -> None:
        self._ensure_writable()
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                f"Deleting mag {mag} failed. There is no mag with this name"
            )
        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) != mag
        ]
        self._save_layer_properties()

    @property
    def name(self) -> str:
        """
        Returns the name of the layer.
        """
        return self._name

    @name.setter
    def name(self, layer_name: str) -> None:
        """
        Renames the layer to `layer_name`. This changes the name of the directory on disk and updates the properties.
        Only layers on local file systems can be renamed.
        """
        from webknossos.dataset.dataset import _validate_layer_name

        if layer_name == self.name:
            return
        self._ensure_metadata_writable()
        if layer_name in self.dataset.layers.keys():
            raise ValueError(
                f"Failed to rename layer {self.name} to {layer_name}: The new name already exists."
            )

        _validate_layer_name(layer_name)

        old_name = self.name
        del self.dataset._layers[self.name]
        self.dataset._layers[layer_name] = self
        self._properties.name = layer_name
        self._name: str = layer_name
        self._save_layer_properties(layer_renaming=(old_name, layer_name))

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError(
                f"Remote layer '{self.name}' is read-only, consider disabling zarr_streaming with RemoteDataset.open(use_zarr_streaming=False)"
            )

    def _apply_server_layer_properties(self) -> None:
        self.dataset._apply_server_dataset_properties()

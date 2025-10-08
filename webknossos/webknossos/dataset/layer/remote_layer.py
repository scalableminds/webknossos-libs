from typing import TYPE_CHECKING

from upath import UPath

from webknossos.dataset_properties import LayerProperties, MagViewProperties

from ...geometry.mag import Mag, MagLike
from ...utils import enrich_path
from .abstract_layer import AbstractLayer
from .layer_to_link import LayerToLink
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

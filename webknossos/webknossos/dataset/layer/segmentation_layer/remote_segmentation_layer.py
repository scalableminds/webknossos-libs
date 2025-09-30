from typing import TYPE_CHECKING

from webknossos.dataset.layer.remote_layer import RemoteLayer
from webknossos.dataset.layer.segmentation_layer import (
    RemoteAttachments,
)
from webknossos.dataset.layer.segmentation_layer.abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)
from webknossos.dataset_properties import SegmentationLayerProperties

if TYPE_CHECKING:
    from webknossos.dataset import RemoteDataset

class RemoteSegmentationLayer(
    AbstractSegmentationLayer[RemoteAttachments], RemoteLayer
):
    def __init__(
        self,
        dataset: "RemoteDataset",
        properties: SegmentationLayerProperties,
        read_only: bool,
    ) -> None:
        super().__init__(dataset, properties, read_only)

    @property
    def attachments_type(self) -> type[RemoteAttachments]:
        return RemoteAttachments

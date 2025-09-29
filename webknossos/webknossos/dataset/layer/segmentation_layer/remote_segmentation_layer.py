from webknossos.dataset import RemoteDataset
from webknossos.dataset.layer.remote_layer import RemoteLayer
from webknossos.dataset.layer.segmentation_layer import (
    AbstractSegmentationLayer,
    RemoteAttachments,
)
from webknossos.dataset_properties import SegmentationLayerProperties


class RemoteSegmentationLayer(
    AbstractSegmentationLayer[RemoteAttachments], RemoteLayer
):
    def __init__(
        self,
        dataset: RemoteDataset,
        properties: SegmentationLayerProperties,
        read_only: bool,
    ) -> None:
        super().__init__(dataset, properties, read_only)

    @property
    def attachments_type(self) -> type[RemoteAttachments]:
        return RemoteAttachments

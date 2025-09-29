from typing import TYPE_CHECKING

from webknossos.dataset.layer.layer import Layer
from webknossos.dataset.layer.segmentation_layer.abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)
from webknossos.dataset.layer.segmentation_layer.attachments import Attachments
from webknossos.dataset_properties import SegmentationLayerProperties

if TYPE_CHECKING:
    from webknossos.dataset import Dataset


class SegmentationLayer(AbstractSegmentationLayer[Attachments], Layer):
    def __init__(
        self,
        dataset: "Dataset",
        properties: SegmentationLayerProperties,
        read_only: bool,
    ):
        super().__init__(dataset, properties, read_only)

    @property
    def attachments_type(self) -> type[Attachments]:
        return Attachments

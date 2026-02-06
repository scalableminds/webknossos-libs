from typing import TYPE_CHECKING

from ....dataset_properties import SegmentationLayerProperties
from ...layer import Layer
from .abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)
from .attachments import Attachments

if TYPE_CHECKING:
    from ... import Dataset


class SegmentationLayer(AbstractSegmentationLayer[Attachments], Layer):
    def __init__(
        self,
        dataset: "Dataset",
        properties: SegmentationLayerProperties,
        read_only: bool,
    ):
        super().__init__(dataset, properties, read_only)

    @property
    def _AttachmentsType(self) -> type[Attachments]:
        return Attachments

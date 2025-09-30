from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from cluster_tools import Executor

from webknossos.dataset.layer.view import View
from webknossos.dataset_properties import (
    SEGMENTATION_CATEGORY,
    LayerCategoryType,
    SegmentationLayerProperties,
)
from webknossos.geometry import Vec3Int

from ..abstract_layer import AbstractLayer
from .attachments import AbstractAttachments

AttachmentsTypeT = TypeVar("AttachmentsTypeT", bound=AbstractAttachments)
if TYPE_CHECKING:
    from webknossos.dataset.abstract_dataset import AbstractDataset


class AbstractSegmentationLayer(AbstractLayer, Generic[AttachmentsTypeT]):
    """A specialized Layer subclass for segmentation data.

    A SegmentationLayer extends the base Layer class with functionality specific
    to segmentation data, such as tracking the largest segment ID. The key
    differences are:

    - Always uses the SEGMENTATION_CATEGORY category type
    - Tracks the largest segment ID present in the data
    - Provides methods for updating the largest segment ID
    - Adds an `attachments` property for managing attachment files

    Attributes:
        largest_segment_id (int | None): Highest segment ID present in data, or None if empty
        category (LayerCategoryType): Always SEGMENTATION_CATEGORY for this class

    Note:
        When creating a new SegmentationLayer, use Dataset.add_layer() rather than
        instantiating directly.
    """

    _properties: SegmentationLayerProperties
    _attachments: AttachmentsTypeT

    def __init__(
        self,
        dataset: "AbstractDataset",
        properties: SegmentationLayerProperties,
        read_only: bool,
    ):
        super().__init__(dataset, properties, read_only)
        self._attachments = self.attachments_type(self, properties.attachments)

    @property
    @abstractmethod
    def attachments_type(self) -> type[AttachmentsTypeT]:
        pass

    @property
    def largest_segment_id(self) -> int | None:
        """Gets the largest segment ID present in the data.

        The largest segment ID is the highest numerical identifier assigned to any
        segment in this layer. This is useful for:
        - Allocating new segment IDs
        - Validating segment ID ranges
        - Optimizing data structures

        Returns:
            int | None: The highest segment ID present, or None if no segments exist
        """
        return self._properties.largest_segment_id

    @largest_segment_id.setter
    def largest_segment_id(self, largest_segment_id: int | None) -> None:
        """Sets the largest segment ID.

        Updates the stored largest segment ID value and persists it to properties.

        Args:
            largest_segment_id (int | None): New largest segment ID value to set.
                Pass None to indicate no segments exist.

        Raises:
            AssertionError: If value is not None and cannot be converted to an integer.
        """

        self._ensure_writable()
        if largest_segment_id is not None and not isinstance(largest_segment_id, int):
            assert largest_segment_id == int(largest_segment_id), (
                f"A non-integer value was passed for largest_segment_id ({largest_segment_id})."
            )
            largest_segment_id = int(largest_segment_id)

        self._properties.largest_segment_id = largest_segment_id
        self._save_layer_properties()

    @property
    def category(self) -> LayerCategoryType:
        return SEGMENTATION_CATEGORY

    @property
    def attachments(self) -> AttachmentsTypeT:
        """Access, add and remove the attachments of this layer.

        Attachments are additional files that can be attached to a segmentation layer.
        They can be used to store additional information, such as meshes, agglomerations, segment indices, cumsums and connectomes.

        Examples:
            ```
            # Add a mesh attachment to the segmentation layer
            layer.attachments.add_mesh(
                mesh_path,
                name="meshfile",
                data_format=AttachmentDataFormat.Zarr3,
            )

            # Access the mesh attachment path
            layer.attachments.meshes[0].path

            # Remove the mesh attachment
            layer.attachments.delete_attachment(layer.attachments.meshes[0])
            ```
        """
        return self._attachments

    def _get_largest_segment_id_maybe(self) -> int | None:
        return self.largest_segment_id

    def _get_largest_segment_id(self, view: View) -> int:
        """Gets the largest segment ID within a view.

        Args:
            view: View of segmentation data

        Returns:
            int: Maximum segment ID value found
        """
        return np.max(view.read(), initial=0)

    def refresh_largest_segment_id(
        self,
        *,
        chunk_shape: Vec3Int | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Updates largest_segment_id based on actual data content.

        Scans through the data to find the highest segment ID value.
        Sets to None if data is empty.

        Args:
            chunk_shape: Shape of chunks for processing
            executor: Executor for parallel processing
        """

        try:
            chunk_results = self.get_finest_mag().map_chunk(
                self._get_largest_segment_id,
                chunk_shape=chunk_shape,
                executor=executor,
                progress_desc="Searching largest segment id",
            )
            self.largest_segment_id = max(chunk_results)
        except ValueError:
            self.largest_segment_id = None

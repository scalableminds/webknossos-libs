import logging
import re
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from upath import UPath

from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    DataFormat,
    LayerCategoryType,
    LayerProperties,
    LayerViewConfiguration,
)
from webknossos.dataset_properties.structuring import (
    MagViewProperties,
)
from webknossos.geometry import NDBoundingBox
from webknossos.geometry.mag import Mag, MagLike
from webknossos.geometry.nd_bounding_box import NormalizedBoundingBox

from ...utils import warn_deprecated
from .view import ArrayException, MagView

if TYPE_CHECKING:
    from ..abstract_dataset import AbstractDataset
    from .segmentation_layer.abstract_segmentation_layer import (
        AbstractSegmentationLayer,
    )


def _validate_layer_name(layer_name: str) -> None:
    if _ALLOWED_LAYER_NAME_REGEX.match(layer_name) is None:
        raise ValueError(
            f"The layer name '{layer_name}' is invalid. It must only contain letters, numbers, underscores, hyphens and dots."
        )


# A layer name is allowed to contain letters, numbers, underscores, hyphens and dots.
# As the begin and the end are anchored, all of the name must match the regex.
# The first regex group ensures that the name does not start with a dot.
_ALLOWED_LAYER_NAME_REGEX = re.compile(r"^[A-Za-z0-9_$@\-]+[A-Za-z0-9_$@\-\.]*$")
# This regex matches any character that is not allowed in a layer name.
_UNALLOWED_LAYER_NAME_CHARS = re.compile(r"[^A-Za-z0-9_$@\-\.]")


class AbstractLayer:
    _dataset: "AbstractDataset"
    _mags: dict[Mag, MagView]

    def __init__(
        self, dataset: "AbstractDataset", properties: LayerProperties, read_only: bool
    ) -> None:
        self._dataset = dataset
        self._apply_properties(properties, read_only)

    def _apply_properties(self, properties: LayerProperties, read_only: bool) -> None:
        # It is possible that the properties on disk do not contain the number of channels.
        # Therefore, the parameter is optional. However at this point, 'num_channels' was already inferred.
        assert "/" not in properties.name and not properties.name.startswith("."), (
            f"The layer name '{properties.name}' is invalid."
        )
        self._name: str = properties.name  # The name is also stored in the properties, but the name is required to get the properties.

        self._dtype = properties.dtype_np
        self._mags = {}
        self._read_only = read_only

        for mag in properties.mags:
            mag_read_only, mag_path = self._determine_read_only_and_path_for_mag(mag)
            self._setup_mag(
                Mag(mag.mag), mag_path, read_only=read_only or mag_read_only
            )

        self._properties.mags = [
            res for res in self._properties.mags if Mag(res.mag) in self._mags
        ]

    @property
    @abstractmethod
    def dataset(self) -> "AbstractDataset":
        pass

    @abstractmethod
    def as_segmentation_layer(self) -> "AbstractSegmentationLayer":
        pass

    @abstractmethod
    def _determine_read_only_and_path_for_mag(
        self, mag_properties: MagViewProperties
    ) -> tuple[bool, UPath]:
        pass

    def _ensure_metadata_writable(self) -> None:
        if self.dataset.read_only:
            raise RuntimeError(
                f"{self.dataset} is read-only, the changes to the metadata of {self} will not be saved!"
            )

    @property
    def _properties(self) -> LayerProperties:
        """Gets the LayerProperties object containing layer attributes.

        Returns:
            LayerProperties: Properties object for this layer

        Note:
            Internal property used to access underlying properties storage.
        """

        return next(
            layer_property
            for layer_property in self.dataset._properties.data_layers
            if layer_property.name == self.name
        )

    def _save_layer_properties(
        self, layer_renaming: tuple[str, str] | None = None
    ) -> None:
        self.dataset._save_dataset_properties(layer_renaming=layer_renaming)

    def _setup_mag(self, mag: Mag, mag_path: UPath, read_only: bool) -> None:
        """Initialize a magnification level when opening the Dataset.

        Does not create storage headers/metadata, e.g. wk_header.

        Args:
            mag: Magnification level to setup
            mag_path: Optional path override for mag data
            read_only: Whether the mag is read_only

        Raises:
            ArrayException: If mag setup fails
        """

        mag_name = mag.to_layer_name()

        self._assert_mag_does_not_exist_yet(mag)
        try:
            self._mags[mag] = MagView(
                self,
                mag,
                mag_path,
                read_only=read_only,
            )
        except ArrayException:
            logging.exception(
                f"Failed to setup magnification {mag_name}, which is specified in the datasource-properties.json:"
            )

    def _assert_mag_does_not_exist_yet(self, mag: MagLike) -> None:
        """Verifies a magnification does not already exist.

        Args:
            mag: Magnification to check

        Raises:
            IndexError: If magnification exists
        """
        if mag in self.mags.keys():
            raise IndexError(
                f"Adding mag {mag} failed. There is already a mag with this name"
            )

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise RuntimeError(f"{self} is read-only, the changes will not be saved!")

    @property
    def bounding_box(self) -> NDBoundingBox:
        """Gets the bounding box encompassing this layer's data.

        Returns:
            NDBoundingBox: Bounding box with layer dimensions
        """

        return self._properties.bounding_box.denormalize()

    @bounding_box.setter
    def bounding_box(self, bbox: NDBoundingBox) -> None:
        """Updates the offset and size of the bounding box of this layer in the properties."""
        self._ensure_metadata_writable()
        assert bbox.topleft.is_positive(), (
            f"Updating the bounding box of layer {self} to {bbox} failed, topleft must not contain negative dimensions."
        )
        bbox = bbox.normalize_axes(self.num_channels)
        self._properties.bounding_box = bbox
        self._save_layer_properties()
        for mag in self.mags.values():
            mag._array.resize(bbox.align_with_mag(mag.mag).in_mag(mag.mag))

    @property
    def normalized_bounding_box(self) -> NormalizedBoundingBox:
        """Gets the bounding box with axes normalized to include the channel dimension.

        Returns:
            NormalizedBoundingBox: Bounding box with channel axis included
        """
        return self._properties.bounding_box

    @property
    def category(self) -> LayerCategoryType:
        """Gets the category type of this layer.

        Returns:
            LayerCategoryType: Layer category (e.g. COLOR_CATEGORY)
        """

        return COLOR_CATEGORY

    @property
    def dtype(self) -> np.dtype:
        """Gets the data type used per channel.

        Returns:
            np.dtype: NumPy data type for individual channels
        """
        return self._dtype

    @property
    def dtype_per_channel(self) -> np.dtype:
        """Deprecated. Use `dtype` instead.
        Gets the data type used per channel.

        Returns:
            np.dtype: NumPy data type for individual channels
        """
        warn_deprecated("dtype_per_channel", "dtype")
        return self._dtype

    @property
    def num_channels(self) -> int:
        """Gets the number of channels in this layer.

        Returns:
            int: Number of channels

        Raises:
            AssertionError: If num_channels is not set in properties
        """

        return self.normalized_bounding_box.size.get("c", 1)

    @property
    def data_format(self) -> DataFormat:
        """Gets the data storage format used by this layer.

        Returns:
            DataFormat: Format used to store data

        Raises:
            AssertionError: If data_format is not set in properties
        """

        assert self._properties.data_format is not None
        return self._properties.data_format

    @property
    def default_view_configuration(self) -> LayerViewConfiguration | None:
        """Gets the default view configuration for this layer.

        Returns:
            LayerViewConfiguration | None: View configuration if set, otherwise None
        """

        return self._properties.default_view_configuration

    @default_view_configuration.setter
    def default_view_configuration(
        self, view_configuration: LayerViewConfiguration
    ) -> None:
        self._ensure_metadata_writable()
        self._properties.default_view_configuration = view_configuration
        self._save_layer_properties()

    @property
    def read_only(self) -> bool:
        """Whether this layer is read-only.

        Returns:
            bool: True if layer is read-only, False if writable
        """
        return self._read_only

    @property
    def mags(self) -> Mapping[Mag, MagView]:
        """
        Getter for dictionary containing all mags.
        """
        return self._mags

    @property
    def name(self) -> str:
        """Gets the name identifier of this layer.

        Returns:
            str: Layer name
        """

        return self._name

    def get_mag(self, mag: MagLike) -> MagView:
        """Gets the MagView for the specified magnification level.

        Returns a view of the data at the requested magnification level. The mag
        parameter can be specified in various formats that will be normalized.

        Args:
            mag: Magnification identifier in multiple formats (int, str, list, etc)

        Returns:
            MagView: View of data at the specified magnification

        Raises:
            IndexError: If specified magnification does not exist
        """
        mag = Mag(mag)
        if mag not in self.mags.keys():
            raise IndexError(
                f"The mag {mag.to_layer_name()} is not a mag of this layer"
            )
        return self.mags[mag]

    def get_finest_mag(self) -> MagView:
        """Gets the MagView with the finest/smallest magnification.

        Returns:
            MagView: View of data at finest available magnification
        """
        return self.get_mag(min(self.mags.keys()))

    def _get_largest_segment_id_maybe(self) -> int | None:
        return None

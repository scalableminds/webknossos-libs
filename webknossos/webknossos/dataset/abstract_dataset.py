import copy
import json
import logging
import re
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike
from upath import UPath

from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DatasetProperties,
    DatasetViewConfiguration,
    LayerProperties,
    SegmentationLayerProperties,
    VoxelSize,
    get_dataset_converter,
)
from webknossos.dataset_properties.dtype_conversion import (
    properties_floating_type_to_python_type,
)
from webknossos.geometry import BoundingBox, NDBoundingBox
from webknossos.utils import warn_deprecated

from .defaults import PROPERTIES_FILE_NAME
from .layer.abstract_layer import AbstractLayer
from .layer.segmentation_layer.abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)

logger = logging.getLogger(__name__)
SUPPORTED_VERSIONS: list[Literal[1]] = [1]
DEFAULT_VERSION: Literal[1] = 1

# Note: The dataset_name part might be outdated. Retrieve the dataset name by requesting the up to date object from the backend.
_DATASET_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/datasets/"
    + r"((?P<dataset_name>[^/]*)-)?(?P<dataset_id>[^/\?#]+)(/(view(#[^?/]*)?)?)?"
    + r"((\?token=(?P<sharing_token>[^#\?]*))[^/]*)?$"
)
_DATASET_DEPRECATED_URL_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/datasets/"
    + r"(?P<organization_id>[^/]*)/(?P<dataset_name>[^/]*)(/(view)?)?"
    + r"(\?token=(?P<sharing_token>[^#]*))?"
)

LayerType = TypeVar("LayerType", bound=AbstractLayer)
SegmentationLayerType = TypeVar(
    "SegmentationLayerType", bound=AbstractSegmentationLayer[Any]
)


def _dtype_maybe(
    dtype: DTypeLike | None, dtype_per_channel: DTypeLike | None
) -> np.dtype | None:
    if dtype is not None and dtype_per_channel is not None:
        raise AttributeError(
            "Cannot add layer. Specifying both 'dtype' and 'dtype_per_channel' is not allowed"
        )
    elif dtype_per_channel is not None:
        warn_deprecated("dtype_per_channel", "dtype")
        return np.dtype(
            properties_floating_type_to_python_type.get(
                dtype_per_channel,  # type: ignore[arg-type]
                dtype_per_channel,  # type: ignore[arg-type]
            )
        )
    elif dtype is not None:
        return np.dtype(
            properties_floating_type_to_python_type.get(
                dtype,  # type: ignore[arg-type]
                dtype,  # type: ignore[arg-type]
            )
        )
    return None


class AbstractDataset(Generic[LayerType, SegmentationLayerType]):
    def __init__(
        self,
        dataset_properties: DatasetProperties,
        read_only: bool,
    ):
        self._init_from_properties(dataset_properties, read_only)

    @property
    @abstractmethod
    def _LayerType(self) -> type[LayerType]:
        pass

    @property
    @abstractmethod
    def _SegmentationLayerType(self) -> type[SegmentationLayerType]:
        pass

    def _init_from_properties(
        self, dataset_properties: DatasetProperties, read_only: bool
    ) -> None:
        assert (
            dataset_properties.version is None
            or dataset_properties.version in SUPPORTED_VERSIONS
        ), f"Unsupported dataset version {dataset_properties.version}"

        self._properties = dataset_properties
        self._last_read_properties = copy.deepcopy(self._properties)
        self._read_only = read_only
        self._layers: dict[str, LayerType] = {}

        # construct self.layers
        for layer_properties in self._properties.data_layers:
            layer = self._initialize_layer_from_properties(
                layer_properties, self.read_only
            )
            self._layers[layer_properties.name] = layer

    def _initialize_layer_from_properties(
        self, properties: LayerProperties, read_only: bool
    ) -> LayerType:
        if properties.category == COLOR_CATEGORY:
            return self._LayerType(self, properties, read_only=read_only)
        elif properties.category == SEGMENTATION_CATEGORY:
            segmentation_layer = self._SegmentationLayerType(
                self, cast(SegmentationLayerProperties, properties), read_only=read_only
            )
            # Make sure we have a valid LayerType and SegmentationLayerType combination.
            assert isinstance(segmentation_layer, self._LayerType), (
                f"Expected a {self._LayerType}, got {type(segmentation_layer)} instead."
            )
            return segmentation_layer
        else:
            raise RuntimeError(
                f"Failed to initialize layer: the specified category ({properties.category}) does not exist."
            )

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise RuntimeError(f"{self} is read-only, the changes will not be saved!")

    @abstractmethod
    def _load_dataset_properties(self) -> DatasetProperties:
        pass

    @abstractmethod
    def _save_dataset_properties_impl(
        self, *, layer_renaming: tuple[str, str] | None = None
    ) -> None:
        pass

    def _save_dataset_properties(
        self,
        *,
        check_existing_properties: bool = True,
        layer_renaming: tuple[str, str] | None = None,
    ) -> None:
        self._ensure_writable()
        if check_existing_properties:
            stored_properties = self._load_dataset_properties()
            try:
                if stored_properties != self._last_read_properties:
                    warnings.warn(
                        "[WARNING] While exporting the dataset's properties, stored properties were found which are "
                        + "newer than the ones that were seen last time. The properties will be overwritten. This is "
                        + "likely happening because multiple processes changed the metadata of this dataset."
                    )
            except ValueError:
                # the __eq__ operator raises a ValueError when two bboxes are not comparable. This is the case when the
                # axes are not the same. During initialization axes are added or moved sometimes.
                warnings.warn(
                    "[WARNING] Properties changed in a way that they are not comparable anymore. Most likely "
                    + "the bounding box naming or axis order changed."
                )
        self._save_dataset_properties_impl(layer_renaming=layer_renaming)
        self._last_read_properties = copy.deepcopy(self._properties)

    @property
    def layers(self) -> Mapping[str, LayerType]:
        """Dictionary containing all layers of this dataset.

        Returns:
            dict[str, Layer]: Dictionary mapping layer names to Layer objects

        Examples:
            ```
            for layer_name, layer in ds.layers.items():
               print(layer_name)
            ```
        """

        return self._layers

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Size of each voxel in nanometers along each dimension (x, y, z).

        Returns:
            tuple[float, float, float]: Size of each voxel in nanometers for x,y,z dimensions

        Examples:
            ```
            vx, vy, vz = ds.voxel_size
            print(f"X resolution is {vx}nm")
            ```
        """

        return self._properties.scale.to_nanometer()

    @property
    def voxel_size_with_unit(self) -> VoxelSize:
        """Size of voxels including unit information.

        Size of each voxel along each dimension (x, y, z), including unit specification.
        The default unit is nanometers.

        Returns:
            VoxelSize: Object containing voxel sizes and their units

        """

        return self._properties.scale

    @property
    def name(self) -> str:
        """Name of this dataset as specified in datasource-properties.json.

        Can be modified to rename the dataset. Changes are persisted to the properties file.

        Returns:
            str: Current dataset name

        Examples:
            ```
            ds.name = "my_renamed_dataset"  # Updates the name in properties file
            ```
        """

        return self._properties.id["name"]

    @name.setter
    def name(self, name: str) -> None:
        self._ensure_writable()
        current_id = self._properties.id
        current_id["name"] = name
        self._properties.id = current_id
        self._save_dataset_properties()

    @property
    def default_view_configuration(self) -> DatasetViewConfiguration | None:
        """Default view configuration for this dataset in webknossos.

        Controls how the dataset is displayed in webknossos when first opened by a user, including position,
        zoom level, rotation etc.

        Returns:
            DatasetViewConfiguration | None: Current view configuration if set

        Examples:
            ```
            ds.default_view_configuration = DatasetViewConfiguration(
                zoom=1.5,
                position=(100, 100, 100)
            )
            ```
        """

        return self._properties.default_view_configuration

    @default_view_configuration.setter
    def default_view_configuration(
        self, view_configuration: DatasetViewConfiguration
    ) -> None:
        self._ensure_writable()
        self._properties.default_view_configuration = view_configuration
        self._save_dataset_properties()  # update properties on disk

    @property
    def read_only(self) -> bool:
        """Whether this dataset is opened in read-only mode.

        When True, operations that would modify the dataset (adding layers, changing properties,
        etc.) are not allowed and will raise RuntimeError.

        Returns:
            bool: True if dataset is read-only, False otherwise
        """

        return self._read_only

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    def get_layer(self, layer_name: str) -> LayerType:
        """Get a specific layer from this dataset.

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            Layer: The requested layer object

        Raises:
            IndexError: If no layer with the given name exists

        Examples:
            ```
            color_layer = ds.get_layer("color")
            seg_layer = ds.get_layer("segmentation")
            ```

        Note:
            Use `layers` property to access all layers at once.
        """
        if layer_name not in self.layers.keys():
            raise IndexError(f"The layer {layer_name} is not a layer of this dataset")
        return self.layers[layer_name]

    def get_segmentation_layers(self) -> list[SegmentationLayerType]:
        """Get all segmentation layers in the dataset.

        Provides access to all layers with category 'segmentation'.
        Useful when a dataset contains multiple segmentation layers.

        Returns:
            list[SegmentationLayer]: List of all segmentation layers in order

        Examples:
            Print all segmentation layer names:
                ```
                for layer in ds.get_segmentation_layers():
                    print(layer.name)
                ```

        Note:
            If you need only a single segmentation layer, consider using
            `get_layer()` with the specific layer name instead.
        """

        return [
            cast(SegmentationLayerType, layer)
            for layer in self.layers.values()
            if layer.category == SEGMENTATION_CATEGORY
        ]

    def get_color_layers(self) -> list[LayerType]:
        """Get all color layers in the dataset.

        Provides access to all layers with category 'color'.
        Useful when a dataset contains multiple color layers.

        Returns:
            list[Layer]: List of all color layers in order

        Examples:
            Print all color layer names:
                ```
                for layer in ds.get_color_layers():
                    print(layer.name)
                ```

        Note:
            If you need only a single color layer, consider using
            `get_layer()` with the specific layer name instead.
        """
        return [
            cast(LayerType, layer)
            for layer in self.layers.values()
            if layer.category == COLOR_CATEGORY
        ]

    def get_segmentation_layer(self, layer_name: str) -> SegmentationLayerType:
        """Get a segmentation layer by name.

        Args:
            layer_name: Name of the layer to get

        Returns:
            SegmentationLayer: The segmentation layer
        """
        return cast(
            SegmentationLayerType, self.get_layer(layer_name).as_segmentation_layer()
        )

    def calculate_bounding_box(self) -> NDBoundingBox:
        """Calculate the enclosing bounding box of all layers.

        Finds the smallest box that contains all data from all layers
        in the dataset.

        Returns:
            NDBoundingBox: Bounding box containing all layer data

        Examples:
            ```
            bbox = ds.calculate_bounding_box()
            print(f"Dataset spans {bbox.size} voxels")
            print(f"Dataset starts at {bbox.topleft}")
            ```
        """

        all_layers = list(self.layers.values())
        if len(all_layers) <= 0:
            return BoundingBox.empty()
        dataset_bbox = all_layers[0].bounding_box
        for layer in all_layers[1:]:
            bbox = layer.bounding_box
            dataset_bbox = dataset_bbox.extended_by(bbox)
        return dataset_bbox

    @staticmethod
    def _load_dataset_properties_from_path(dataset_path: UPath) -> DatasetProperties:
        try:
            data = json.loads((dataset_path / PROPERTIES_FILE_NAME).read_bytes())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot read dataset at {dataset_path}. datasource-properties.json file not found."
            )
        properties = get_dataset_converter().structure(data, DatasetProperties)
        return properties

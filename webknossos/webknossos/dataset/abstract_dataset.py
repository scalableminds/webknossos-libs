import copy
import inspect
import json
import logging
import re
import warnings
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from upath import UPath

from webknossos.client.api_client.errors import UnexpectedStatusError
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
from webknossos.geometry import BoundingBox, NDBoundingBox

from ..utils import warn_deprecated
from .defaults import PROPERTIES_FILE_NAME
from .layer.abstract_layer import AbstractLayer
from .layer.segmentation_layer.abstract_segmentation_layer import (
    AbstractSegmentationLayer,
)
from .remote_folder import RemoteFolder

if TYPE_CHECKING:
    from ..client.context import webknossos_context
    from .dataset import RemoteDataset

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
    def _save_dataset_properties_impl(self) -> None:
        pass

    def _save_dataset_properties(self, check_existing_properties: bool = True) -> None:
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
        self._save_dataset_properties_impl()
        self._last_read_properties = copy.deepcopy(self._properties)

    @property
    def layers(self) -> dict[str, LayerType]:
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
    def get_remote_datasets(
        *,
        organization_id: str | None = None,
        tags: str | Sequence[str] | None = None,
        name: str | None = None,
        folder_id: RemoteFolder | str | None = None,
    ) -> Mapping[str, "RemoteDataset"]:
        warn_deprecated("Dataset.get_remote_datasets", "RemoteDataset.list")
        from webknossos import RemoteDataset

        return RemoteDataset.list(organization_id, tags, name, folder_id)

    @classmethod
    def _disambiguate_remote(
        cls,
        dataset_name: str,
        organization_id: str,
    ) -> str:
        from ..client.context import _get_context
        from webknossos import RemoteDataset
        current_context = _get_context()
        possible_ids = list(
            RemoteDataset.list(
                name=dataset_name, organization_id=organization_id
            ).keys()
        )
        if len(possible_ids) == 0:
            try:
                dataset_id = current_context.api_client_with_auth.dataset_id_from_name(
                    directory_name=dataset_name, organization_id=organization_id
                )
                possible_ids.append(dataset_id)
            except UnexpectedStatusError:
                raise ValueError(
                    f"Dataset with name {dataset_name} not found in organization {organization_id}"
                )
        elif len(possible_ids) > 1:
            logger.warning(
                f"There are several datasets with same name '{dataset_name}' available online. Opened dataset with ID {possible_ids[0]}. "
                "If this is not the correct dataset, please provide the dataset ID. You can get the dataset IDs "
                "of your datasets with `Dataset.get_remote_datasets(name=<dataset_name>)."
            )
        return possible_ids[0]

    @classmethod
    def _parse_remote(
        cls,
        dataset_name_or_url: str | None = None,
        organization_id: str | None = None,
        sharing_token: str | None = None,
        webknossos_url: str | None = None,
        dataset_id: str | None = None,
    ) -> tuple["webknossos_context", str, str | None]:
        """Parses the given arguments to
        * context_manager that should be entered,
        * dataset_id,
        """
        from ..client._resolve_short_link import resolve_short_link
        from ..client.context import _get_context, webknossos_context

        caller = inspect.stack()[1].function
        current_context = _get_context()

        if dataset_id is None:
            assert dataset_name_or_url is not None, (
                f"Please supply either a dataset_id or a dataset name or url to Dataset.{caller}()."
            )
            dataset_name_or_url = resolve_short_link(dataset_name_or_url)

            match = _DATASET_URL_REGEX.match(dataset_name_or_url)
            deprecated_match = _DATASET_DEPRECATED_URL_REGEX.match(dataset_name_or_url)
            if match is not None:
                assert (
                    organization_id is None
                    and sharing_token is None
                    and webknossos_url is None
                ), (
                    f"When Dataset.{caller}() is called with an url, "
                    + f"e.g. Dataset.{caller}('https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view'), "
                    + "organization_id, sharing_token and webknossos_url must not be set."
                )
                dataset_id = match.group("dataset_id")
                sharing_token = match.group("sharing_token")
                webknossos_url = match.group("webknossos_url")
                assert dataset_id is not None
            elif deprecated_match is not None:
                assert (
                    organization_id is None
                    and sharing_token is None
                    and webknossos_url is None
                ), (
                    f"When Dataset.{caller}() is called with an url, "
                    + f"e.g. Dataset.{caller}('https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view'), "
                    + "organization_id, sharing_token and webknossos_url must not be set."
                )
                dataset_name = deprecated_match.group("dataset_name")
                organization_id = deprecated_match.group("organization_id")
                sharing_token = deprecated_match.group("sharing_token")
                webknossos_url = deprecated_match.group("webknossos_url")

                assert organization_id is not None
                assert dataset_name is not None

                dataset_id = cls._disambiguate_remote(dataset_name, organization_id)
            else:
                dataset_name = dataset_name_or_url
                organization_id = organization_id or current_context.organization_id

                dataset_id = cls._disambiguate_remote(dataset_name, organization_id)

        if webknossos_url is None:
            webknossos_url = current_context.url
        webknossos_url = webknossos_url.rstrip("/")
        context_manager = webknossos_context(
            webknossos_url, token=sharing_token or current_context.token
        )
        if webknossos_url != current_context.url:
            if sharing_token is None:
                warnings.warn(
                    f"[INFO] The supplied url {webknossos_url} does not match your current context {current_context.url}. "
                    + f"Using no token, only public datasets can used with Dataset.{caller}(). "
                    + "Please see https://docs.webknossos.org/api/webknossos/client/context.html to adapt the URL and token."
                )
                context_manager = webknossos_context(webknossos_url, None)
        return (context_manager, dataset_id, sharing_token)

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

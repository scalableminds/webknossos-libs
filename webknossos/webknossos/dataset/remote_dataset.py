import copy
import inspect
import logging
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Union

import attr
import numpy as np
from boltons.typeutils import make_sentinel
from cluster_tools import Executor
from numpy.typing import DTypeLike
from upath import UPath

from webknossos.client import webknossos_context
from webknossos.client.api_client.models import (
    ApiDataset,
    ApiDatasetExploreAndAddRemote,
    ApiLayerRenaming,
    ApiMetadata,
    ApiUnusableDataSource,
)
from webknossos.dataset._metadata import DatasetMetadata
from webknossos.dataset.abstract_dataset import (
    _DATASET_DEPRECATED_URL_REGEX,
    _DATASET_URL_REGEX,
    AbstractDataset,
)
from webknossos.dataset.layer import RemoteLayer, RemoteSegmentationLayer, Zarr3Config
from webknossos.dataset.layer.abstract_layer import (
    _dtype_per_layer_to_dtype_per_channel,
    _normalize_dtype_per_channel,
    _normalize_dtype_per_layer,
    _validate_layer_name,
)
from webknossos.dataset.sampling_modes import SamplingModes
from webknossos.dataset_properties import (
    COLOR_CATEGORY,
    SEGMENTATION_CATEGORY,
    DataFormat,
    DatasetProperties,
    LayerCategoryType,
    LayerProperties,
    SegmentationLayerProperties,
)
from webknossos.geometry import BoundingBox, NDBoundingBox, Vec3Int
from webknossos.geometry.mag import Mag, MagLike
from webknossos.utils import infer_metadata_type, warn_deprecated

from ..client.api_client.errors import UnexpectedStatusError
from ..dataset_properties.structuring import (
    _properties_floating_type_to_python_type,
)
from ..ssl_context import SSL_CONTEXT
from .defaults import DEFAULT_BIT_DEPTH, DEFAULT_DATA_FORMAT
from .layer.abstract_layer import _dtype_per_channel_to_element_class
from .remote_dataset_registry import RemoteDatasetRegistry
from .remote_folder import RemoteFolder
from .transfer_mode import TransferMode

logger = logging.getLogger(__name__)
_UNSET = make_sentinel("UNSET", var_name="_UNSET")


if TYPE_CHECKING:
    from webknossos.administration.user import Team
    from webknossos.dataset import Dataset


class RemoteDataset(AbstractDataset[RemoteLayer, RemoteSegmentationLayer]):
    """A representation of a dataset managed by a WEBKNOSSOS server.

    This class is returned from `RemoteDataset.open()` and provides read-only access to
    image data streamed from the webknossos server. It uses the same interface as `Dataset`
    but additionally allows metadata manipulation through properties.
    In case of zarr streaming, an even smaller subset of metadata manipulation is possible.

    Properties:
        metadata: Dataset metadata as key-value pairs
        name: Human readable name
        description: Dataset description
        tags: Dataset tags
        is_public: Whether dataset is public
        sharing_token: Dataset sharing token
        allowed_teams: Teams with dataset access
        folder: Dataset folder location

    Examples:
        Opening a remote dataset with organization ID:
        ```
        ds = RemoteDataset.open("my_dataset", "org_id")
        ```

        Opening with dataset URL:
        ```
        ds = RemoteDataset.open("https://webknossos.org/datasets/org/dataset/view")
        ```

        Setting metadata:
        ```
        ds.metadata = {"key": "value", "tags": ["tag1", "tag2"]}
        ds.name = "My_Dataset"
        ds.allowed_teams = [Team.get_by_name("Lab_A")]
        ```

    Note:
        Do not instantiate directly, use `RemoteDataset.open()` or `RemoteDataset.open_remote` instead.
    """

    def __init__(
        self,
        zarr_streaming_path: UPath | None,
        dataset_properties: DatasetProperties | None,
        dataset_id: str,
        annotation_id: str | None,
        context: webknossos_context,
        read_only: bool,
    ) -> None:
        """Initialize a remote dataset instance.

        Args:
            zarr_streaming_path: Path to the zarr streaming directory
            dataset_properties: Properties of the remote dataset
            dataset_id: dataset id of the remote dataset
            context: Context manager for WEBKNOSSOS connection

        Raises:
            FileNotFoundError: If dataset cannot be opened as zarr format and no metadata exists

        Note:
            Do not call this constructor directly, use RemoteDataset.open() instead.
            This class provides access to remote WEBKNOSSOS datasets with additional metadata manipulation.
        """
        assert (zarr_streaming_path is not None) != (dataset_properties is not None), (
            "Either zarr_streaming_path or dataset_properties must be set, but not both."
        )
        self.zarr_streaming_path = zarr_streaming_path
        self._use_zarr_streaming = zarr_streaming_path is not None
        if self._use_zarr_streaming:
            dataset_properties = self._load_dataset_properties()

        assert dataset_properties is not None
        super().__init__(dataset_properties, read_only)

        self._dataset_id = dataset_id
        self._annotation_id = annotation_id
        self._context = context

    @classmethod
    def open(
        cls,
        dataset_name_or_url: str | None = None,
        organization_id: str | None = None,
        sharing_token: str | None = None,
        webknossos_url: str | None = None,
        dataset_id: str | None = None,
        annotation_id_or_url: str | None = None,
        use_zarr_streaming: bool = True,
        read_only: bool = False,
    ) -> "RemoteDataset":
        """Opens a remote webknossos dataset. Image data is accessed via network requests.
        Dataset metadata such as allowed teams or the sharing token can be read and set
        via the respective `RemoteDataset` properties.

        Args:
            dataset_name_or_url: Either dataset name or full URL to dataset view, e.g.
                https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view
            organization_id: Optional organization ID if using dataset name. Can be found [here](https://webknossos.org/auth/token)
            sharing_token: Optional sharing token for dataset access
            webknossos_url: Optional custom webknossos URL, defaults to context URL, usually https://webknossos.org
            dataset_id: Optional unique ID of the dataset
            annotation_id_or_url: Optional unique ID or URL of the annotation to stream the data from the annotation.
            use_zarr_streaming: Whether to use zarr streaming

        Returns:
            RemoteDataset: Dataset instance for remote access

        Examples:
            ```
            ds = RemoteDataset.open("`https://webknossos.org/datasets/scalable_minds/l4_sample_dev/view`")
            ```

        Note:
            If supplying an URL, organization_id, webknossos_url and sharing_token
            must not be set.
        """

        if annotation_id_or_url is not None:
            assert use_zarr_streaming, (
                "Annotations are only supported with zarr streaming"
            )

        from ..client.context import _get_context

        (context_manager, dataset_id, annotation_id, sharing_token) = cls._parse_remote(
            dataset_name_or_url,
            organization_id,
            sharing_token,
            webknossos_url,
            dataset_id,
            annotation_id_or_url,
        )

        with context_manager:
            wk_context = _get_context()
            token = sharing_token or wk_context.token
            api_dataset_info = wk_context.api_client.dataset_info(
                dataset_id=dataset_id, sharing_token=sharing_token
            )
            datastore_url = api_dataset_info.data_store.url
            url_prefix = wk_context.get_datastore_api_client(datastore_url).url_prefix

            if use_zarr_streaming:
                if annotation_id is not None:
                    zarr_path = UPath(
                        f"{url_prefix}/annotations/zarr/{annotation_id}/",
                        headers={} if token is None else {"X-Auth-Token": token},
                        ssl=SSL_CONTEXT,
                    )
                else:
                    zarr_path = UPath(
                        f"{url_prefix}/zarr/{dataset_id}/",
                        headers={} if token is None else {"X-Auth-Token": token},
                        ssl=SSL_CONTEXT,
                    )
                return cls(
                    zarr_path,
                    None,
                    dataset_id,
                    annotation_id,
                    context_manager,
                    read_only=read_only,
                )
            else:
                if isinstance(api_dataset_info.data_source, ApiUnusableDataSource):
                    raise RuntimeError(
                        f"The dataset {dataset_id} is unusable {api_dataset_info.data_source.status}"
                    )

                return cls(
                    None,
                    api_dataset_info.data_source,
                    dataset_id,
                    annotation_id,
                    context_manager,
                    read_only,
                )

    def _initialize_layer_from_properties(
        self, properties: "LayerProperties", read_only: bool
    ) -> RemoteLayer:
        # When using zarr streaming, layers are read only.
        read_only = self._use_zarr_streaming
        return super()._initialize_layer_from_properties(properties, read_only)

    @property
    def _LayerType(self) -> type[RemoteLayer]:
        return RemoteLayer

    @property
    def _SegmentationLayerType(self) -> type[RemoteSegmentationLayer]:
        return RemoteSegmentationLayer

    def _load_dataset_properties(self) -> DatasetProperties:
        if self._use_zarr_streaming:
            assert self.zarr_streaming_path is not None
            return self._load_dataset_properties_from_path(self.zarr_streaming_path)

        api_dataset_info = self._get_dataset_info()
        assert isinstance(api_dataset_info.data_source, DatasetProperties)
        return api_dataset_info.data_source

    def _apply_server_dataset_properties(self) -> None:
        self._properties = self._load_dataset_properties()
        self._last_read_properties = copy.deepcopy(self._properties)

    def _save_dataset_properties_impl(
        self, layer_renaming: tuple[str, str] | None = None
    ) -> None:
        """
        Exports the current dataset properties to the server.
        Note that some edits will not be accepted by the server.
        The client-side RemoteDataset is reinitialized to the new server state.
        Does not work with zarr streaming, as the remote datasource-properties.json is not writable.
        """
        from ..client.context import _get_api_client

        if self._use_zarr_streaming:
            # reset the dataset properties to the server state
            self._apply_server_dataset_properties()
            raise RuntimeError("zarr streaming does not support updating this property")

        layer_renamings = []
        if layer_renaming is not None:
            layer_renamings.append(
                ApiLayerRenaming(old_name=layer_renaming[0], new_name=layer_renaming[1])
            )

        with self._context:
            client = _get_api_client()
            client.dataset_update(
                dataset_id=self._dataset_id,
                dataset_updates={
                    "dataSource": self._properties,
                    "layerRenamings": layer_renamings,
                },
            )
            self._apply_server_dataset_properties()

    def __repr__(self) -> str:
        return f"RemoteDataset({repr(self.url)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.dataset_id == other.dataset_id
        else:
            return False

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def annotation_id(self) -> str | None:
        return self._annotation_id

    @property
    def url(self) -> str:
        """URL to access this dataset in webknossos.

        Constructs the full URL to the dataset in the webknossos web interface.

        Returns:
            str: Full dataset URL including organization and dataset name

        Examples:
            ```
            print(ds.url) # 'https://webknossos.org/datasets/my_org/my_dataset'
            ```
        """

        from ..client.context import _get_context

        with self._context:
            wk_url = _get_context().url
        return f"{wk_url}/datasets/{self._dataset_id}"

    @property
    def created(self) -> str:
        """Creation date of the dataset.

        Returns:
            str: Date and time when dataset was created
        """

        return datetime.fromtimestamp(self._get_dataset_info().created / 1000).strftime(
            "%B %d, %Y %I:%M:%S"
        )

    @property
    def used_storage_bytes(self) -> int:
        """The amount of storage used by the dataset on the WEBKNOSSOS server. Note that 0 may indicate that the data is either not stored on the WEBKNOSSOS server directly, or that the storage usage was not yet scanned.

        Returns:
            int: The amount of storage used by the dataset in bytes.
        """
        return self._get_dataset_info().used_storage_bytes or 0

    def _get_dataset_info(self) -> ApiDataset:
        from ..client.context import _get_api_client

        with self._context:
            client = _get_api_client()
            return client.dataset_info(dataset_id=self._dataset_id)

    def _update_dataset_info(
        self,
        name: str = _UNSET,
        description: str | None = _UNSET,
        is_public: bool = _UNSET,
        folder_id: str = _UNSET,
        tags: list[str] = _UNSET,
        metadata: list[ApiMetadata] | None = _UNSET,
    ) -> None:
        self._ensure_writable()
        from ..client.context import _get_api_client

        # Atm, the wk backend needs to get previous parameters passed
        # (this is a race-condition with parallel updates).

        dataset_updates: dict[str, Any] = {}

        if name is not _UNSET:
            dataset_updates["name"] = name
        if description is not _UNSET:
            dataset_updates["description"] = description
        if tags is not _UNSET:
            dataset_updates["tags"] = tags
        if is_public is not _UNSET:
            dataset_updates["isPublic"] = is_public
        if folder_id is not _UNSET:
            dataset_updates["folderId"] = folder_id
        if metadata is not _UNSET:
            dataset_updates["metadata"] = metadata

        with self._context:
            client = _get_api_client()
            client.dataset_update(
                dataset_id=self._dataset_id, dataset_updates=dataset_updates
            )

    @property
    def metadata(self) -> DatasetMetadata:
        """Get or set metadata key-value pairs for the dataset.

        The metadata can contain strings, numbers, and lists of strings as values.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            DatasetMetadata: Current metadata key-value pairs

        Examples:
            ```
            ds.metadata = {
                "species": "mouse",
                "age_days": 42,
                "tags": ["verified", "published"]
            }
            print(ds.metadata["species"])
            ```
        """

        return DatasetMetadata(f"{self._dataset_id}")

    @metadata.setter
    def metadata(
        self,
        metadata: dict[str, str | int | float | Sequence[str]] | DatasetMetadata | None,
    ) -> None:
        if metadata is not None:
            api_metadata = [
                ApiMetadata(key=k, type=infer_metadata_type(v), value=v)
                for k, v in metadata.items()
            ]
        self._update_dataset_info(metadata=api_metadata)

    @property
    def name(self) -> str:
        """The human-readable name for the dataset in the webknossos interface.

        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            str | None: Current display name if set, None otherwise

        Examples:
            ```
            remote_ds.name = "Mouse Brain Sample A"
            ```
        """

        return self._get_dataset_info().name

    @name.setter
    def name(self, name: str) -> None:
        self._update_dataset_info(name=name)

    @property
    def display_name(self) -> str:
        """Deprecated, please use `name`.
        The human-readable name for the dataset in the webknossos interface.

        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            str | None: Current display name if set, None otherwise

        Examples:
            ```
            remote_ds.name = "Mouse Brain Sample A"
            ```
        """
        warn_deprecated("display_name", "name")

        return self.name

    @display_name.setter
    def display_name(self, name: str) -> None:
        warn_deprecated("display_name", "name")

        self.name = name

    @property
    def description(self) -> str | None:
        """Free-text description of the dataset.

        Can be edited with markdown formatting. Changes are immediately synchronized
        with WEBKNOSSOS.

        Returns:
            str | None: Current description if set, None otherwise

        Examples:
            ```
            ds.description = "Dataset acquired on *June 1st*"
            ds.description = None  # Remove description
            ```
        """

        return self._get_dataset_info().description

    @description.setter
    def description(self, description: str | None) -> None:
        self._update_dataset_info(description=description)

    @description.deleter
    def description(self) -> None:
        self.description = None

    @property
    def tags(self) -> tuple[str, ...]:
        """User-assigned tags for organizing and filtering datasets.

        Tags allow categorizing and filtering datasets in the webknossos dashboard interface.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            tuple[str, ...]: Currently assigned tags, in string tuple form

        Examples:
            ```
            ds.tags = ["verified", "published"]
            print(ds.tags)  # ('verified', 'published')
            ds.tags = []  # Remove all tags
            ```
        """

        return tuple(self._get_dataset_info().tags)

    @tags.setter
    def tags(self, tags: Sequence[str]) -> None:
        self._update_dataset_info(tags=list(tags))

    @property
    def is_public(self) -> bool:
        """Control whether the dataset is publicly accessible.

        When True, anyone can view the dataset without logging in to WEBKNOSSOS.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            bool: True if dataset is public, False if private

        Examples:
            ```
            ds.is_public = True
            ds.is_public = False
            print("Public" if ds.is_public else "Private")  # Private
            ```
        """

        return bool(self._get_dataset_info().is_public)

    @is_public.setter
    def is_public(self, is_public: bool) -> None:
        self._update_dataset_info(is_public=is_public)

    @property
    def sharing_token(self) -> str:
        """Get a new token for sharing access to this dataset.

        Each call generates a fresh token that allows viewing the dataset without logging in.
        The token can be appended to dataset URLs as a query parameter.

        Returns:
            str: Fresh sharing token for dataset access

        Examples:
            ```
            token = ds.sharing_token
            url = f"{ds.url}?token={token}"
            print("Share this link:", url)
            ```

        Note:
            - A new token is generated on each access
            - The token provides read-only access
            - Anyone with the token can view the dataset
        """

        from ..client.context import _get_api_client

        with self._context:
            api_sharing_token = _get_api_client().dataset_sharing_token(
                dataset_id=self._dataset_id
            )
            return api_sharing_token.sharing_token

    @property
    def allowed_teams(self) -> tuple["Team", ...]:
        """Teams that are allowed to access this dataset.

        Controls which teams have read access to view and use this dataset.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            tuple[Team, ...]: Teams currently having access

        Examples:
            ```
            from webknossos import Team
            team = Team.get_by_name("Lab_A")
            ds.allowed_teams = [team]
            print([t.name for t in ds.allowed_teams])

            # Give access to multiple teams:
            ds.allowed_teams = [
                Team.get_by_name("Lab_A"),
                Team.get_by_name("Lab_B")
            ]
            ```

        Note:
            - Teams must be from the same organization as the dataset
            - Can be set using Team objects or team ID strings
            - An empty list makes the dataset private
        """

        from ..administration.team import Team

        return tuple(
            Team(id=i.id, name=i.name, organization_id=i.organization)
            for i in self._get_dataset_info().allowed_teams
        )

    @allowed_teams.setter
    def allowed_teams(self, allowed_teams: Sequence[Union[str, "Team"]]) -> None:
        """Assign the teams that are allowed to access the dataset. Specify the teams like this `[Team.get_by_name("Lab_A"), ...]`."""
        self._ensure_writable()
        from ..administration.team import Team
        from ..client.context import _get_api_client

        team_ids = [i.id if isinstance(i, Team) else i for i in allowed_teams]

        with self._context:
            client = _get_api_client()
            client.dataset_update_teams(dataset_id=self._dataset_id, team_ids=team_ids)

    @property
    def folder(self) -> RemoteFolder:
        """The (virtual) folder containing this dataset in WEBKNOSSOS.

        Represents the folder location in the WEBKNOSSOS UI folder structure.
        Can be changed to move the dataset to a different folder.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            RemoteFolder: Current folder containing the dataset

        Examples:
            ```
            folder = RemoteFolder.get_by_path("Datasets/Published")
            ds.folder = folder
            print(ds.folder.path) # 'Datasets/Published'
            ```

        """

        return RemoteFolder.get_by_id(self._get_dataset_info().folder_id)

    @folder.setter
    def folder(self, folder: RemoteFolder) -> None:
        """Move the dataset to a folder. Specify the folder like this `RemoteFolder.get_by_path("Datasets/Folder_A")`."""
        self._update_dataset_info(folder_id=folder.id)

    def download(
        self,
        sharing_token: str | None = None,
        bbox: BoundingBox | None = None,
        layers: list[str] | str | None = None,
        mags: list[Mag] | None = None,
        path: PathLike | UPath | str | None = None,
        exist_ok: bool = False,
    ) -> "Dataset":
        """Downloads a dataset and returns the Dataset instance.
        * `sharing_token` may be supplied if a dataset name was used and can specify a sharing token.
        * `bbox`, `layers`, and `mags` specify which parts of the dataset to download.
          If nothing is specified the whole image, all layers, and all mags are downloaded respectively.
        * `path` and `exist_ok` specify where to save the downloaded dataset and whether to overwrite
          if the `path` exists.
        """
        from ..client._download_dataset import download_dataset

        if isinstance(layers, str):
            layers = [layers]
        return download_dataset(
            dataset_id=self.dataset_id,
            sharing_token=sharing_token,
            bbox=bbox,
            layers=layers,
            mags=mags,
            path=UPath(path) if path is not None else None,
            exist_ok=exist_ok,
        )

    def download_mesh(
        self,
        segment_id: int,
        output_dir: PathLike | UPath | str,
        layer_name: str | None = None,
        mesh_file_name: str | None = None,
        datastore_url: str | None = None,
        lod: int = 0,
        mapping_name: str | None = None,
        mapping_type: Literal["agglomerate", "json"] | None = None,
        mag: MagLike | None = None,
        seed_position: Vec3Int | None = None,
        token: str | None = None,
        sharing_token: str | None = None,
    ) -> UPath:
        warn_deprecated(
            "RemoteDataset.download_mesh", "RemoteSegmentationLayer.download"
        )
        if layer_name is None:
            segmentation_layers = self.get_segmentation_layers()
            if len(segmentation_layers) != 1:
                raise ValueError(
                    "When you attempt to download a mesh without a layer_name, there must be exactly one segmentation layer."
                )
            segmentation_layer = segmentation_layers[0]
        else:
            segmentation_layer = self.get_segmentation_layer(layer_name)
        return segmentation_layer.download_mesh(
            segment_id,
            output_dir,
            mesh_file_name,
            datastore_url,
            lod,
            mapping_name,
            mapping_type,
            mag,
            seed_position,
            token,
            sharing_token,
        )

    def delete_layer(self, layer_name: str) -> None:
        self._ensure_writable()

        if layer_name not in self.layers.keys():
            raise IndexError(
                f"Removing layer {layer_name} failed. There is no layer with this name"
            )
        del self._layers[layer_name]
        self._properties.data_layers = [
            layer for layer in self._properties.data_layers if layer.name != layer_name
        ]
        self._save_dataset_properties()

    def add_layer(
        self,
        layer_name: str,
        category: LayerCategoryType,
        *,
        dtype_per_layer: DTypeLike | None = None,
        dtype_per_channel: DTypeLike | None = None,
        num_channels: int | None = None,
        data_format: str | DataFormat = DEFAULT_DATA_FORMAT,
        bounding_box: NDBoundingBox | None = None,
        **kwargs: Any,
    ) -> RemoteLayer:
        self._ensure_writable()

        _validate_layer_name(layer_name)

        if num_channels is None:
            num_channels = 1

        if dtype_per_layer is not None and dtype_per_channel is not None:
            raise AttributeError(
                "Cannot add layer. Specifying both 'dtype_per_layer' and 'dtype_per_channel' is not allowed"
            )
        elif dtype_per_channel is not None:
            dtype_per_channel = _properties_floating_type_to_python_type.get(
                dtype_per_channel,  # type: ignore[arg-type]
                dtype_per_channel,  # type: ignore[arg-type]
            )
            dtype_per_channel = _normalize_dtype_per_channel(dtype_per_channel)  # type: ignore[arg-type]
        elif dtype_per_layer is not None:
            warn_deprecated("dtype_per_layer", "dtype_per_channel")
            dtype_per_layer = _properties_floating_type_to_python_type.get(
                dtype_per_layer,  # type: ignore[arg-type]
                dtype_per_layer,  # type: ignore[arg-type]
            )
            dtype_per_layer = _normalize_dtype_per_layer(dtype_per_layer)  # type: ignore[arg-type]
            dtype_per_channel = _dtype_per_layer_to_dtype_per_channel(
                dtype_per_layer, num_channels
            )
        else:
            dtype_per_channel = np.dtype("uint" + str(DEFAULT_BIT_DEPTH))

        if layer_name in self.layers.keys():
            raise IndexError(
                f"Adding layer {layer_name} failed. There is already a layer with this name"
            )

        assert data_format != DataFormat.WKW, (
            "Cannot create WKW layers in remote datasets. Use `data_format='zarr'`."
        )

        layer_properties = LayerProperties(
            name=layer_name,
            category=category,
            bounding_box=bounding_box or BoundingBox((0, 0, 0), (0, 0, 0)),
            element_class=_dtype_per_channel_to_element_class(
                dtype_per_channel, num_channels
            ),
            mags=[],
            num_channels=num_channels,
            data_format=DataFormat(data_format),
        )

        if category == COLOR_CATEGORY:
            self._properties.data_layers += [layer_properties]
            self._layers[layer_name] = RemoteLayer(
                self, layer_properties, read_only=False
            )
        elif category == SEGMENTATION_CATEGORY:
            segmentation_layer_properties: SegmentationLayerProperties = (
                SegmentationLayerProperties(
                    **(
                        attr.asdict(layer_properties, recurse=False)
                    ),  # use all attributes from LayerProperties
                    largest_segment_id=kwargs.get("largest_segment_id"),
                )
            )
            if "mappings" in kwargs:
                segmentation_layer_properties.mappings = kwargs["mappings"]
            self._properties.data_layers += [segmentation_layer_properties]
            self._layers[layer_name] = RemoteSegmentationLayer(
                self, segmentation_layer_properties, read_only=False
            )
        else:
            raise RuntimeError(
                f"Failed to add layer ({layer_name}) because of invalid category ({category}). The supported categories are '{COLOR_CATEGORY}' and '{SEGMENTATION_CATEGORY}'"
            )

        self._save_dataset_properties()
        return self.layers[layer_name]

    @classmethod
    def list(
        cls,
        organization_id: str | None = None,
        tags: str | Sequence[str] | None = None,
        name: str | None = None,
        folder_id: RemoteFolder | str | None = None,
    ) -> Mapping[str, "RemoteDataset"]:
        """Get all available datasets from the WEBKNOSSOS server.

        Returns a mapping of dataset ids to lazy-initialized RemoteDataset objects for all
        datasets visible to the specified organization or current user. Datasets can be further filtered by tags, name or folder.

        Args:
            organization_id: Optional organization to get datasets from. Defaults to
                organization of logged in user.
            tags: Optional tag(s) to filter datasets by. Can be a single tag string or
                sequence of tags. Only returns datasets with all specified tags.
            name: Optional name to filter datasets by. Only returns datasets with
                matching name.
            folder: Optional folder to filter datasets by. Only returns datasets in
                the specified folder.

        Returns:
            Mapping[str, RemoteDataset]: Dict mapping dataset ids to RemoteDataset objects

        Examples:
            List all available datasets:
            ```
            datasets = RemoteDataset.list()
            print(sorted(datasets.keys()))
            ```

            Get datasets for specific organization:
            ```
            org_datasets = RemoteDataset.list()("my_organization")
            ds = org_datasets["dataset_name"]
            ```

            Filter datasets by tag:
            ```
            published = RemoteDataset.list(tags="published")
            tagged = RemoteDataset.list(tags=["tag1", "tag2"])
            ```

            Filter datasets by name:
            ```
            fun_datasets = RemoteDataset.list(name="MyFunDataset")
            ```

        Note:
            RemoteDataset objects are initialized lazily when accessed for the first time.
            The mapping object provides a fast way to list and look up available datasets.
        """
        if isinstance(folder_id, RemoteFolder):
            folder_id = folder_id.id

        return RemoteDatasetRegistry(
            name=name, organization_id=organization_id, tags=tags, folder_id=folder_id
        )

    @classmethod
    def _disambiguate_remote(
        cls,
        dataset_name: str,
        organization_id: str,
    ) -> str:
        from webknossos import RemoteDataset

        from ..client.context import _get_api_client

        client = _get_api_client()
        possible_ids = list(
            RemoteDataset.list(
                name=dataset_name, organization_id=organization_id
            ).keys()
        )
        if len(possible_ids) == 0:
            try:
                dataset_id = client.dataset_id_from_name(
                    directory_name=dataset_name, organization_id=organization_id
                )
                possible_ids.append(dataset_id)
            except UnexpectedStatusError as e:
                raise ValueError(
                    f"Dataset with name {dataset_name} not found in organization {organization_id}"
                ) from e
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
        annotation_id_or_url: str | None = None,
    ) -> tuple["webknossos_context", str, str | None, str | None]:
        """Parses the given arguments to
        * context_manager that should be entered,
        * dataset_id,
        """
        from .. import Annotation
        from ..client._resolve_short_link import resolve_short_link
        from ..client.context import _get_context, webknossos_context

        caller = inspect.stack()[1].function
        current_context = _get_context()

        if annotation_id_or_url is not None:
            annotation = Annotation.download(
                annotation_id_or_url,
                webknossos_url=webknossos_url,
                skip_volume_data=True,
            )
            if dataset_id is not None:
                assert dataset_id == annotation.dataset_id, (
                    f"The annotation id {annotation.annotation_id} is not from the dataset with id {dataset_id}."
                )
            dataset_id = annotation.dataset_id
            annotation_id = annotation.annotation_id
        else:
            annotation_id = None

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
        return (context_manager, dataset_id, annotation_id, sharing_token)

    @classmethod
    def trigger_reload_in_datastore(
        cls,
        dataset_name_or_url: str | None = None,
        organization_id: str | None = None,
        webknossos_url: str | None = None,
        dataset_id: str | None = None,
        organization: str | None = None,  # deprecated, use organization_id instead
        datastore_url: str | None = None,
        token: str | None = None,  # deprecated, use a webknossos context instead
    ) -> None:
        """Trigger a manual reload of the dataset's properties.

        For manually uploaded datasets, properties are normally updated automatically
        after a few minutes. This method forces an immediate reload.

        This is typically only needed after manual changes to the dataset's files.
        Cannot be used for local datasets.

        Args:
            dataset_name_or_url: Name or URL of dataset to reload
            dataset_id: ID of dataset to reload
            organization_id: Organization ID where dataset is located
            datastore_url: Optional URL to the datastore
            webknossos_url: Optional URL to the webknossos server

        Examples:
            ```
            # Force reload after manual file changes
            Dataset.trigger_reload_in_datastore(
                "my_dataset",
                "organization_id"
            )
            ```
        """

        from ..client._upload_dataset import _cached_get_upload_datastore
        from ..client.context import _get_context

        if token is not None:
            warn_deprecated("parameter: token", "a webknossos context")

        if organization is not None:
            warn_deprecated("organization", "organization_id")
            if organization_id is None:
                organization_id = organization
            else:
                raise ValueError(
                    "Both organization and organization_id were provided. Only one is allowed."
                )

        (context_manager, dataset_id, _, _) = cls._parse_remote(
            dataset_name_or_url,
            organization_id,
            token,
            webknossos_url,
            dataset_id,
        )

        with context_manager:
            context = _get_context()
            datastore_url = datastore_url or _cached_get_upload_datastore(context)
            organization_id = organization_id or context.organization_id

            datastore_client = context.get_datastore_api_client(datastore_url)
            datastore_client.dataset_trigger_reload(
                organization_id=organization_id, dataset_id=dataset_id
            )

    @classmethod
    def explore_and_add_remote(
        cls, dataset_uri: str | PathLike | UPath, dataset_name: str, folder_path: str
    ) -> "RemoteDataset":
        """Explore and add an external dataset as a remote dataset.

        Adds a dataset from an external location (e.g. S3, Google Cloud Storage, or HTTPs) to WEBKNOSSOS by inspecting
        its layout and metadata without copying the data.

        Args:
            dataset_uri: URI pointing to the remote dataset location
            dataset_name: Name to register dataset under in WEBKNOSSOS
            folder_path: Path in WEBKNOSSOS folder structure where dataset should appear

        Returns:
            RemoteDataset: The newly added dataset accessible via WEBKNOSSOS

        Examples:
            ```
            remote = Dataset.explore_and_add_remote(
                "s3://bucket/dataset",
                "my_dataset",
                "Datasets/Research"
            )
            ```

        Note:
            The dataset files must be accessible from the WEBKNOSSOS server
            for this to work. The data will be streamed through webknossos from the source.
        """
        from ..client.context import _get_api_client

        client = _get_api_client()
        dataset = ApiDatasetExploreAndAddRemote(
            UPath(dataset_uri).resolve().as_uri(), dataset_name, folder_path
        )
        dataset_id = client.dataset_explore_and_add_remote(dataset=dataset)

        return cls.open(dataset_id=dataset_id)

    def downsample(
        self,
        *,
        sampling_mode: SamplingModes = SamplingModes.ANISOTROPIC,
        coarsest_mag: Mag | None = None,
        interpolation_mode: str = "default",
        compress: bool | Zarr3Config = True,
        transfer_mode: TransferMode = TransferMode.COPY,
        common_storage_path_prefix: str | None = None,
        overwrite_pending: bool = True,
        executor: Executor | None = None,
    ) -> None:
        """Generate downsampled magnifications for all layers.

        Creates lower resolution versions (coarser magnifications) of all layers that are not
        yet downsampled, up to the specified coarsest magnification.

        Args:
            sampling_mode: Strategy for downsampling (e.g. ANISOTROPIC, MAX)
            coarsest_mag: Optional maximum/coarsest magnification to generate
            interpolation_mode: Interpolation method to use. Defaults to "default" (= "mode" for segmentation, "median" for color).
            compress: Whether to compress generated magnifications. For Zarr3 datasets, codec configuration and chunk key encoding may also be supplied. Defaults to True.
            transfer_mode (TransferMode). How new mags are transferred to the remote or local storage. Defaults to COPY
            common_storage_path_prefix (str | None): Optional path prefix used when transfer_mode is either COPY or MOVE_AND_SYMLINK
                                        to select one of the available WEBKNOSSOS storages.
            overwrite_pending (bool). If there are already pending/unfinished committed mags on the server, overwrite them. Defaults to True
            executor: Optional executor for parallel processing

        Raises:
            RuntimeError: If dataset is read-only

        Examples:
            Basic downsampling:
                ```
                ds.downsample()
                ```

            With custom parameters:
                ```
                ds.downsample(
                    sampling_mode=SamplingModes.ANISOTROPIC,
                    coarsest_mag=Mag(8),
                )
                ```

        Note:
            - ANISOTROPIC sampling creates anisotropic downsampling until dataset is isotropic
            - Other modes like MAX, CONSTANT etc create regular downsampling patterns
            - If magnifications already exist they will not be regenerated
        """
        for layer in self.layers.values():
            layer.downsample(
                coarsest_mag=coarsest_mag,
                sampling_mode=sampling_mode,
                interpolation_mode=interpolation_mode,
                compress=compress,
                transfer_mode=transfer_mode,
                common_storage_path_prefix=common_storage_path_prefix,
                overwrite_pending=overwrite_pending,
                executor=executor,
            )

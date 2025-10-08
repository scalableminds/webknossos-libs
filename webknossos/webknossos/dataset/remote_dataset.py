import logging
from collections.abc import Sequence
from datetime import datetime
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Union

from boltons.typeutils import make_sentinel
from upath import UPath

from webknossos.client import webknossos_context
from webknossos.client.api_client.models import (
    ApiDataset,
    ApiDatasetExploreAndAddRemote,
    ApiMetadata,
)
from webknossos.dataset._metadata import DatasetMetadata
from webknossos.dataset.abstract_dataset import AbstractDataset
from webknossos.dataset.layer import RemoteLayer, RemoteSegmentationLayer
from webknossos.dataset_properties import (
    DatasetProperties,
)
from webknossos.geometry import Vec3Int
from webknossos.geometry.mag import MagLike
from webknossos.utils import infer_metadata_type, warn_deprecated

from .remote_folder import RemoteFolder

logger = logging.getLogger(__name__)
_UNSET = make_sentinel("UNSET", var_name="_UNSET")

if TYPE_CHECKING:
    from webknossos.administration.user import Team


class RemoteDataset(AbstractDataset[RemoteLayer, RemoteSegmentationLayer]):
    """A representation of a dataset managed by a WEBKNOSSOS server.

    This class is returned from `Dataset.open_remote()` and provides read-only access to
    image data streamed from the webknossos server. It uses the same interface as `Dataset`
    but additionally allows metadata manipulation through properties.

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
        ds = Dataset.open_remote("my_dataset", "org_id")
        ```

        Opening with dataset URL:
        ```
        ds = Dataset.open_remote("https://webknossos.org/datasets/org/dataset/view")
        ```

        Setting metadata:
        ```
        ds.metadata = {"key": "value", "tags": ["tag1", "tag2"]}
        ds.name = "My_Dataset"
        ds.allowed_teams = [Team.get_by_name("Lab_A")]
        ```

    Note:
        Do not instantiate directly, use `Dataset.open_remote()` or `RemoteDataset.open_remote` instead.
    """

    def __init__(
        self,
        zarr_streaming_path: UPath | None,
        dataset_properties: DatasetProperties | None,
        dataset_id: str,
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
            Do not call this constructor directly, use Dataset.open_remote() instead.
            This class provides access to remote WEBKNOSSOS datasets with additional metadata manipulation.
        """
        assert (zarr_streaming_path is not None) != (dataset_properties is not None), (
            "Either zarr_streaming_path or dataset_properties must be set, but not both."
        )
        self.zarr_streaming_path = zarr_streaming_path
        self._use_zarr_streaming = zarr_streaming_path is not None
        if self._use_zarr_streaming:
            assert read_only, "zarr streaming is only supported in read-only mode"
            dataset_properties = self._load_dataset_properties()

        assert dataset_properties is not None
        super().__init__(dataset_properties, read_only)

        self._dataset_id = dataset_id
        self._context = context

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

    def _save_dataset_properties_impl(self) -> None:
        """
        Exports the current dataset properties to the server.
        Note that some edits will not be accepted by the server.
        The client-side RemoteDataset is reinitialized to the new server state.
        """
        from ..client.context import _get_api_client

        with self._context:
            client = _get_api_client()
            client.dataset_update(
                dataset_id=self._dataset_id,
                dataset_updates={"dataSource": self._properties},
            )
            data_source = self._load_dataset_properties()

            self._init_from_properties(data_source, read_only=self.read_only)

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

    @classmethod
    def explore_and_add_remote(
        cls, dataset_uri: str | PathLike, dataset_name: str, folder_path: str
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
        from ..client.context import _get_context

        context = _get_context()
        dataset = ApiDatasetExploreAndAddRemote(
            UPath(dataset_uri).resolve().as_uri(), dataset_name, folder_path
        )
        context.api_client_with_auth.dataset_explore_and_add_remote(dataset=dataset)

        return cls.open_remote(
            dataset_name_or_url=dataset_name, organization_id=context.organization_id
        )

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

    def download_mesh(
        self,
        segment_id: int,
        output_dir: PathLike | str,
        layer_name: str | None = None,
        mesh_file_name: str | None = None,
        datastore_url: str | None = None,
        lod: int = 0,
        mapping_name: str | None = None,
        mapping_type: Literal["agglomerate", "json"] | None = None,
        mag: MagLike | None = None,
        seed_position: Vec3Int | None = None,
        token: str | None = None,
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
        )

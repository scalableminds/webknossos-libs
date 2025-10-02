from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Union

import attr

from ..administration.user import Team
from ..client.api_client.models import ApiFolderUpdate, ApiFolderWithParent, ApiMetadata
from ..utils import infer_metadata_type
from ._metadata import FolderMetadata

if TYPE_CHECKING:
    from .dataset import RemoteDataset


def _get_folder_path(
    folder: ApiFolderWithParent,
    all_folders: Iterable[ApiFolderWithParent],
) -> str:
    if folder.parent is None:
        return folder.name
    else:
        return f"{_get_folder_path(next(f for f in all_folders if f.id == folder.parent), all_folders)}/{folder.name}"


@attr.define
class RemoteFolder:
    """This class is used to access and edit metadata of a folder on the webknossos server."""

    id: str
    _name: str

    def add_subfolder(self, name: str) -> "RemoteFolder":
        """Adds a new folder with the specified name."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder = client.folder_add(folder_name=name, parent_id=self.id)
        return RemoteFolder(name=folder.name, id=folder.id)

    @classmethod
    def get_by_id(cls, folder_id: str) -> "RemoteFolder":
        """Returns the folder specified by the passed id."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder_tree_response: list[ApiFolderWithParent] = client.folder_tree()

        for folder_info in folder_tree_response:
            if folder_info.id == folder_id:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {folder_id}.")

    @classmethod
    def get_by_path(cls, path: str) -> "RemoteFolder":
        """Returns the folder specified by the passed path.
        Separate multiple folder names with a slash."""
        from ..client.context import _get_api_client

        path = path.rstrip("/")
        client = _get_api_client(enforce_auth=True)
        folder_tree_response: list[ApiFolderWithParent] = client.folder_tree()

        for folder_info in folder_tree_response:
            folder_path = _get_folder_path(folder_info, folder_tree_response)
            if folder_path == path:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {path}.")

    @classmethod
    def get_root(cls) -> "RemoteFolder":
        """Returns the root folder of the current organization."""

        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        root_folder = client.folder_root()
        return cls(name=root_folder.name, id=root_folder.id)

    def get_datasets(self) -> Mapping[str, "RemoteDataset"]:
        """Returns all datasets in this folder."""

        from .dataset import Dataset

        return Dataset.get_remote_datasets(folder_id=self.id)

    def get_subfolders(self) -> tuple["RemoteFolder", ...]:
        """Returns all subfolders in this folder."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder_tree_response: list[ApiFolderWithParent] = client.folder_tree()

        return tuple(
            RemoteFolder(name=folder_info.name, id=folder_info.id)
            for folder_info in folder_tree_response
            if folder_info.parent == self.id
        )

    @property
    def metadata(self) -> FolderMetadata:
        return FolderMetadata(self.id)

    @metadata.setter
    def metadata(
        self, metadata: dict[str, str | int | float | Sequence[str]] | None
    ) -> None:
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder = client.folder_get(folder_id=self.id)
        if metadata is not None:
            api_metadata = [
                ApiMetadata(key=k, type=infer_metadata_type(v), value=v)
                for k, v in metadata.items()
            ]
        new_folder = ApiFolderUpdate(
            name=folder.name,
            allowed_teams=[t.id for t in folder.allowed_teams],
            metadata=api_metadata,
        )
        client.folder_update(folder_id=self.id, folder=new_folder)

    def move_to(self, new_parent: "RemoteFolder") -> None:
        """Move the folder to a new parent folder."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        client.folder_move(folder_id=self.id, new_parent_id=new_parent.id)

    @property
    def allowed_teams(self) -> tuple[Team, ...]:
        """Teams that are allowed to access this folder.
        Controls which teams have read access to view and use this folder.
        Changes are immediately synchronized with WEBKNOSSOS.

        Returns:
            tuple[Team, ...]: Teams currently having access
        """
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        return tuple(
            Team(id=t.id, name=t.name, organization_id=t.organization)
            for t in client.folder_get(folder_id=self.id).allowed_teams
        )

    @allowed_teams.setter
    def allowed_teams(self, allowed_teams: Sequence[Union[str, "Team"]]) -> None:
        """Assign the teams that are allowed to access the dataset. Specify the teams like this `[Team.get_by_name("Lab_A"), ...]`."""
        from ..client.context import _get_api_client

        team_ids = [i.id if isinstance(i, Team) else i for i in allowed_teams]

        client = _get_api_client(enforce_auth=True)
        folder = client.folder_get(folder_id=self.id)
        new_folder = ApiFolderUpdate(
            name=folder.name,
            allowed_teams=team_ids,
            metadata=folder.metadata,
        )
        client.folder_update(folder_id=self.id, folder=new_folder)

    @property
    def name(self) -> str:
        """The human-readable name for the folder in the WEBKNOSSOS interface.
        Changes are immediately synchronized with WEBKNOSSOS.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Updates the human-readable name for the folder in the WEBKNOSSOS interface.
        Changes are immediately synchronized with WEBKNOSSOS.
        """
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder = client.folder_get(folder_id=self.id)
        new_folder = ApiFolderUpdate(
            name=name,
            allowed_teams=[t.id for t in folder.allowed_teams],
            metadata=folder.metadata,
        )
        client.folder_update(folder_id=self.id, folder=new_folder)
        self._name = name

    def delete(self) -> None:
        """Deletes the folder."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        client.folder_delete(folder_id=self.id)

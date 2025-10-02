from collections.abc import Iterable, Sequence

import attr

from ..client.api_client.models import ApiFolder, ApiFolderWithParent, ApiMetadata
from ..utils import infer_metadata_type
from ._metadata import FolderMetadata


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
    name: str

    def add_subfolder(self, name: str) -> "RemoteFolder":
        """Adds a new folder with the specified name."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder = client.folder_add("/folders", folder_name=name, parent_id=self.id)
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

    @property
    def metadata(self) -> FolderMetadata:
        return FolderMetadata(self.id)

    @metadata.setter
    def metadata(
        self, metadata: dict[str, str | int | float | Sequence[str]] | None
    ) -> None:
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder = client._get_json(f"/folders/{self.id}", ApiFolder)
        if metadata is not None:
            api_metadata = [
                ApiMetadata(key=k, type=infer_metadata_type(v), value=v)
                for k, v in metadata.items()
            ]
        folder.metadata = api_metadata
        client._put_json(f"/folders/{self.id}", folder)

    def move(self, new_parent: "str | RemoteFolder | None") -> None:
        """Move the folder to a new parent folder."""
        from ..client.context import _get_api_client

        if isinstance(new_parent, str):
            new_parent = RemoteFolder.get_by_path(new_parent)
        elif new_parent is None:
            new_parent = RemoteFolder.get_root()

        client = _get_api_client(enforce_auth=True)
        client.folder_move(folder_id=self.id, parent_id=new_parent.id)

    @property
    def allowed_teams(self) -> list["Team"]:
        """Returns the teams that are allowed to access this folder."""
        pass

    @allowed_teams.setter
    def allowed_teams(self, allowed_teams: Sequence[Union[str, "Team"]]) -> None:
        pass

    @property
    def name(self) -> str:
        """Name of the folder."""
        pass

    @name.setter
    def name(self, name: str) -> None:
        pass

    def delete(self) -> None:
        """Deletes the folder."""
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        client.folder_delete(folder_id=self.id)

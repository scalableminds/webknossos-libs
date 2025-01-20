from typing import Dict, Iterable, List, Optional, Sequence, Union

import attr

from webknossos.dataset._metadata import FolderMetadata
from webknossos.utils import infer_metadata_type

from ..client.api_client.models import ApiFolder, ApiFolderWithParent, ApiMetadata


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

    @classmethod
    def get_by_id(cls, folder_id: str) -> "RemoteFolder":
        from ..client.context import _get_api_client

        client = _get_api_client(enforce_auth=True)
        folder_tree_response: List[ApiFolderWithParent] = client.folder_tree()

        for folder_info in folder_tree_response:
            if folder_info.id == folder_id:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {folder_id}.")

    @classmethod
    def get_by_path(cls, path: str) -> "RemoteFolder":
        from ..client.context import _get_api_client

        path = path.rstrip("/")
        client = _get_api_client(enforce_auth=True)
        folder_tree_response: List[ApiFolderWithParent] = client.folder_tree()

        for folder_info in folder_tree_response:
            folder_path = _get_folder_path(folder_info, folder_tree_response)
            if folder_path == path:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {path}.")

    @property
    def metadata(self) -> FolderMetadata:
        return FolderMetadata(self.id)

    @metadata.setter
    def metadata(
        self, metadata: Optional[Dict[str, Union[str, int, float, Sequence[str]]]]
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

from typing import Iterable, List

import attr

from ..client.api_client.models import ApiFolderWithParent


def _get_folder_path(
    folder: ApiFolderWithParent,
    all_folders: Iterable[ApiFolderWithParent],
) -> str:
    if folder.parent is None:
        return folder.name
    else:
        return f"{_get_folder_path(next(f for f in all_folders if f.id == folder.parent), all_folders)}/{folder.name}"


@attr.frozen
class RemoteFolder:
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

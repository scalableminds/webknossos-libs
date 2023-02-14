from typing import TYPE_CHECKING, Iterable

import attr

if TYPE_CHECKING:
    from webknossos.client._generated.models.folder_tree_response_200_item import (
        FolderTreeResponse200Item,
    )


def _get_folder_path(
    folder: "FolderTreeResponse200Item",
    all_folders: Iterable["FolderTreeResponse200Item"],
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
        from webknossos.client._generated.api.default import folder_tree
        from webknossos.client.context import _get_generated_client

        client = _get_generated_client(enforce_auth=True)

        response = folder_tree.sync(client=client)
        assert response is not None

        for folder_info in response:
            if folder_info.id == folder_id:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {folder_id}.")

    @classmethod
    def get_by_path(cls, path: str) -> "RemoteFolder":
        from webknossos.client._generated.api.default import folder_tree
        from webknossos.client.context import _get_generated_client

        client = _get_generated_client(enforce_auth=True)

        response = folder_tree.sync(client=client)
        assert response is not None

        for folder_info in response:
            folder_path = _get_folder_path(folder_info, response)
            if folder_path == path:
                return cls(name=folder_info.name, id=folder_info.id)

        raise KeyError(f"Could not find folder {path}.")

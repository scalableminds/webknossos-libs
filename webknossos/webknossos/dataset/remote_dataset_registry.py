from typing import TYPE_CHECKING, Optional, Sequence, TypeVar, Union

from ..utils import LazyReadOnlyDict

if TYPE_CHECKING:
    from .dataset import RemoteDataset  # noqa: F401 imported but unused


K = TypeVar("K")  # key
V = TypeVar("V")  # value
C = TypeVar("C")  # cache


class RemoteDatasetRegistry(LazyReadOnlyDict[str, "RemoteDataset"]):
    """Dict-like class mapping dataset ids to `RemoteDataset` instances."""

    def __init__(
        self,
        name: Optional[str],
        organization_id: Optional[str],
        tags: Optional[Union[str, Sequence[str]]],
        folder_id: Optional[str],
    ) -> None:
        from ..administration.user import User
        from ..client.context import _get_api_client
        from .dataset import Dataset

        client = _get_api_client(enforce_auth=True)

        if organization_id is None:
            organization_id = User.get_current_user().organization_id

        if isinstance(tags, str):
            tags = [tags]

        dataset_infos = client.dataset_list(
            is_active=True,
            organization_id=organization_id,
            name=name,
            folder_id=folder_id,
        )

        datasets_ids = []

        for dataset_info in dataset_infos:
            tags_match = tags is None or any(tag in tags for tag in dataset_info.tags)
            name_match = name is None or name == dataset_info.name
            if tags_match and name_match:
                datasets_ids.append(dataset_info.id)

        super().__init__(
            entries=dict(zip(datasets_ids, datasets_ids)),
            func=lambda dataset_id: Dataset.open_remote(dataset_id),
        )

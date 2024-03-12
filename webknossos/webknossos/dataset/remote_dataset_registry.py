from typing import TYPE_CHECKING, Optional, Sequence, TypeVar, Union

from ..utils import LazyReadOnlyDict

if TYPE_CHECKING:
    from .dataset import RemoteDataset  # noqa: F401 imported but unused


K = TypeVar("K")  # key
V = TypeVar("V")  # value
C = TypeVar("C")  # cache


class RemoteDatasetRegistry(LazyReadOnlyDict[str, "RemoteDataset"]):
    """Dict-like class mapping dataset names to `RemoteDataset` instances."""

    def __init__(
        self,
        organization_id: Optional[str],
        tags: Optional[Union[str, Sequence[str]]],
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
            is_active=True, organization_name=organization_id
        )

        datasets_names = []

        for dataset_info in dataset_infos:
            tags_match = tags is None or any(tag in tags for tag in dataset_info.tags)
            if tags_match:
                datasets_names.append(dataset_info.name)

        super().__init__(
            entries=dict(zip(datasets_names, datasets_names)),
            func=lambda name: Dataset.open_remote(name, organization_id),
        )

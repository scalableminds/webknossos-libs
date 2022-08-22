from typing import TYPE_CHECKING, Optional, Sequence, TypeVar, Union

from webknossos.utils import LazyReadOnlyDict

if TYPE_CHECKING:
    from webknossos.dataset.dataset import RemoteDataset


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
        from webknossos.administration.user import User
        from webknossos.client._generated.api.default import dataset_list
        from webknossos.client.context import _get_generated_client
        from webknossos.dataset.dataset import Dataset

        client = _get_generated_client(enforce_auth=True)

        if organization_id is None:
            organization_id = User.get_current_user().organization_id

        if isinstance(tags, str):
            tags = [tags]

        response = dataset_list.sync(
            is_active=True,
            # is_unreported=False,  # this is included in is_active=True
            organization_name=organization_id,
            client=client,
        )
        assert response is not None

        ds_names = []

        for ds_info in response:
            tags_match = tags is None or any(tag in tags for tag in ds_info.tags)
            if tags_match:
                ds_names.append(ds_info.name)

        super().__init__(
            entries=dict(zip(ds_names, ds_names)),
            func=lambda name: Dataset.open_remote(name, organization_id),
        )

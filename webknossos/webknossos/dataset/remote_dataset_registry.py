from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Mapping, TypeVar

from boltons.dictutils import FrozenDict

from webknossos.utils import LazyReadOnlyDict

if TYPE_CHECKING:
    from webknossos.dataset.dataset import RemoteDataset


K = TypeVar("K")  # key
V = TypeVar("V")  # value
C = TypeVar("C")  # cache


class RemoteOrganizationDatasetRegistry(LazyReadOnlyDict[str, "RemoteDataset"]):
    """Dict-like class mapping dataset names to `RemoteDataset` instances."""

    by_display_name: Mapping[str, "RemoteDataset"]
    by_tag: Mapping[str, List["RemoteDataset"]]

    def __init__(
        self,
        organization_id: str,
        names: List[str],
        by_display_name: Dict[str, str],
        by_tag: Dict[str, List[str]],
    ) -> None:
        from webknossos.dataset.dataset import Dataset

        super().__init__(
            entries=dict(zip(names, names)),
            func=lambda name: Dataset.open_remote(name, organization_id),
        )
        self.by_display_name = LazyReadOnlyDict(
            entries=by_display_name,
            func=lambda name: Dataset.open_remote(name, organization_id),
        )
        self.by_tag = LazyReadOnlyDict(
            entries=by_tag,
            func=lambda names: [
                Dataset.open_remote(name, organization_id) for name in names
            ],
        )


class RemoteDatasetRegistry(
    FrozenDict, Mapping[str, RemoteOrganizationDatasetRegistry]
):
    """Dict-like class mapping organization ids to `RemoteOrganizationDatasetRegistry` instances."""

    def __init__(
        self,
    ) -> None:
        from webknossos.client._generated.api.default import dataset_list
        from webknossos.client.context import _get_generated_client

        client = _get_generated_client(enforce_auth=True)
        response = dataset_list.sync(client=client)
        assert response is not None

        names_per_org: DefaultDict[str, List[str]] = defaultdict(list)
        by_display_name_per_org: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
        by_tag_per_org: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(lambda: defaultdict(list))  # type: ignore[call-overload]

        for ds_info in response:
            names_per_org[ds_info.owning_organization].append(ds_info.name)
            if ds_info.display_name:
                by_display_name_per_org[ds_info.owning_organization][
                    ds_info.display_name
                ] = ds_info.name
            for tag in ds_info.tags:
                by_tag_per_org[ds_info.owning_organization][tag].append(ds_info.name)

        super().__init__(
            (
                org,
                RemoteOrganizationDatasetRegistry(
                    org,
                    names=names_per_org[org],
                    by_display_name=by_display_name_per_org[org],
                    by_tag=by_tag_per_org[org],
                ),
            )
            for org in names_per_org
        )

from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset_info_response_200_allowed_teams_cumulative_item import (
        DatasetInfoResponse200AllowedTeamsCumulativeItem,
    )
    from ..models.dataset_info_response_200_allowed_teams_item import (
        DatasetInfoResponse200AllowedTeamsItem,
    )
    from ..models.dataset_info_response_200_data_source import (
        DatasetInfoResponse200DataSource,
    )
    from ..models.dataset_info_response_200_data_store import (
        DatasetInfoResponse200DataStore,
    )


T = TypeVar("T", bound="DatasetInfoResponse200")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200:
    """
    Attributes:
        name (str):
        data_source (DatasetInfoResponse200DataSource):
        data_store (DatasetInfoResponse200DataStore):
        allowed_teams (List['DatasetInfoResponse200AllowedTeamsItem']):
        is_active (int):
        is_public (int):
        description (str):
        display_name (str):
        created (int):
        tags (List[Any]):
        folder_id (str):
        owning_organization (Union[Unset, str]):
        allowed_teams_cumulative (Union[Unset, List['DatasetInfoResponse200AllowedTeamsCumulativeItem']]):
        is_editable (Union[Unset, int]):
        last_used_by_user (Union[Unset, int]):
        logo_url (Union[Unset, str]):
        sorting_key (Union[Unset, int]):
        details (Union[Unset, str]):
        is_unreported (Union[Unset, int]):
        jobs_enabled (Union[Unset, int]):
        publication (Union[Unset, str]):
    """

    name: str
    data_source: "DatasetInfoResponse200DataSource"
    data_store: "DatasetInfoResponse200DataStore"
    allowed_teams: List["DatasetInfoResponse200AllowedTeamsItem"]
    is_active: int
    is_public: int
    description: str
    display_name: str
    created: int
    tags: List[Any]
    folder_id: str
    owning_organization: Union[Unset, str] = UNSET
    allowed_teams_cumulative: Union[
        Unset, List["DatasetInfoResponse200AllowedTeamsCumulativeItem"]
    ] = UNSET
    is_editable: Union[Unset, int] = UNSET
    last_used_by_user: Union[Unset, int] = UNSET
    logo_url: Union[Unset, str] = UNSET
    sorting_key: Union[Unset, int] = UNSET
    details: Union[Unset, str] = UNSET
    is_unreported: Union[Unset, int] = UNSET
    jobs_enabled: Union[Unset, int] = UNSET
    publication: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        data_source = self.data_source.to_dict()

        data_store = self.data_store.to_dict()

        allowed_teams = []
        for allowed_teams_item_data in self.allowed_teams:
            allowed_teams_item = allowed_teams_item_data.to_dict()

            allowed_teams.append(allowed_teams_item)

        is_active = self.is_active
        is_public = self.is_public
        description = self.description
        display_name = self.display_name
        created = self.created
        tags = self.tags

        folder_id = self.folder_id
        owning_organization = self.owning_organization
        allowed_teams_cumulative: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.allowed_teams_cumulative, Unset):
            allowed_teams_cumulative = []
            for allowed_teams_cumulative_item_data in self.allowed_teams_cumulative:
                allowed_teams_cumulative_item = (
                    allowed_teams_cumulative_item_data.to_dict()
                )

                allowed_teams_cumulative.append(allowed_teams_cumulative_item)

        is_editable = self.is_editable
        last_used_by_user = self.last_used_by_user
        logo_url = self.logo_url
        sorting_key = self.sorting_key
        details = self.details
        is_unreported = self.is_unreported
        jobs_enabled = self.jobs_enabled
        publication = self.publication

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "dataSource": data_source,
                "dataStore": data_store,
                "allowedTeams": allowed_teams,
                "isActive": is_active,
                "isPublic": is_public,
                "description": description,
                "displayName": display_name,
                "created": created,
                "tags": tags,
                "folderId": folder_id,
            }
        )
        if owning_organization is not UNSET:
            field_dict["owningOrganization"] = owning_organization
        if allowed_teams_cumulative is not UNSET:
            field_dict["allowedTeamsCumulative"] = allowed_teams_cumulative
        if is_editable is not UNSET:
            field_dict["isEditable"] = is_editable
        if last_used_by_user is not UNSET:
            field_dict["lastUsedByUser"] = last_used_by_user
        if logo_url is not UNSET:
            field_dict["logoUrl"] = logo_url
        if sorting_key is not UNSET:
            field_dict["sortingKey"] = sorting_key
        if details is not UNSET:
            field_dict["details"] = details
        if is_unreported is not UNSET:
            field_dict["isUnreported"] = is_unreported
        if jobs_enabled is not UNSET:
            field_dict["jobsEnabled"] = jobs_enabled
        if publication is not UNSET:
            field_dict["publication"] = publication

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.dataset_info_response_200_allowed_teams_cumulative_item import (
            DatasetInfoResponse200AllowedTeamsCumulativeItem,
        )
        from ..models.dataset_info_response_200_allowed_teams_item import (
            DatasetInfoResponse200AllowedTeamsItem,
        )
        from ..models.dataset_info_response_200_data_source import (
            DatasetInfoResponse200DataSource,
        )
        from ..models.dataset_info_response_200_data_store import (
            DatasetInfoResponse200DataStore,
        )

        d = src_dict.copy()
        name = d.pop("name")

        data_source = DatasetInfoResponse200DataSource.from_dict(d.pop("dataSource"))

        data_store = DatasetInfoResponse200DataStore.from_dict(d.pop("dataStore"))

        allowed_teams = []
        _allowed_teams = d.pop("allowedTeams")
        for allowed_teams_item_data in _allowed_teams:
            allowed_teams_item = DatasetInfoResponse200AllowedTeamsItem.from_dict(
                allowed_teams_item_data
            )

            allowed_teams.append(allowed_teams_item)

        is_active = d.pop("isActive")

        is_public = d.pop("isPublic")

        description = d.pop("description")

        display_name = d.pop("displayName")

        created = d.pop("created")

        tags = cast(List[Any], d.pop("tags"))

        folder_id = d.pop("folderId")

        owning_organization = d.pop("owningOrganization", UNSET)

        allowed_teams_cumulative = []
        _allowed_teams_cumulative = d.pop("allowedTeamsCumulative", UNSET)
        for allowed_teams_cumulative_item_data in _allowed_teams_cumulative or []:
            allowed_teams_cumulative_item = (
                DatasetInfoResponse200AllowedTeamsCumulativeItem.from_dict(
                    allowed_teams_cumulative_item_data
                )
            )

            allowed_teams_cumulative.append(allowed_teams_cumulative_item)

        is_editable = d.pop("isEditable", UNSET)

        last_used_by_user = d.pop("lastUsedByUser", UNSET)

        logo_url = d.pop("logoUrl", UNSET)

        sorting_key = d.pop("sortingKey", UNSET)

        details = d.pop("details", UNSET)

        is_unreported = d.pop("isUnreported", UNSET)

        jobs_enabled = d.pop("jobsEnabled", UNSET)

        publication = d.pop("publication", UNSET)

        dataset_info_response_200 = cls(
            name=name,
            data_source=data_source,
            data_store=data_store,
            allowed_teams=allowed_teams,
            is_active=is_active,
            is_public=is_public,
            description=description,
            display_name=display_name,
            created=created,
            tags=tags,
            folder_id=folder_id,
            owning_organization=owning_organization,
            allowed_teams_cumulative=allowed_teams_cumulative,
            is_editable=is_editable,
            last_used_by_user=last_used_by_user,
            logo_url=logo_url,
            sorting_key=sorting_key,
            details=details,
            is_unreported=is_unreported,
            jobs_enabled=jobs_enabled,
            publication=publication,
        )

        dataset_info_response_200.additional_properties = d
        return dataset_info_response_200

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

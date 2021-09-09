from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.dataset_info_response_200_data_source import (
    DatasetInfoResponse200DataSource,
)
from ..models.dataset_info_response_200_data_store import (
    DatasetInfoResponse200DataStore,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetInfoResponse200")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200:
    """ """

    name: Union[Unset, str] = UNSET
    data_source: Union[Unset, DatasetInfoResponse200DataSource] = UNSET
    data_store: Union[Unset, DatasetInfoResponse200DataStore] = UNSET
    owning_organization: Union[Unset, str] = UNSET
    allowed_teams: Union[Unset, List[Any]] = UNSET
    is_active: Union[Unset, int] = UNSET
    is_public: Union[Unset, int] = UNSET
    description: Union[Unset, str] = UNSET
    display_name: Union[Unset, str] = UNSET
    created: Union[Unset, int] = UNSET
    is_editable: Union[Unset, int] = UNSET
    last_used_by_user: Union[Unset, int] = UNSET
    logo_url: Union[Unset, str] = UNSET
    sorting_key: Union[Unset, int] = UNSET
    details: Union[Unset, str] = UNSET
    publication: Union[Unset, str] = UNSET
    is_unreported: Union[Unset, int] = UNSET
    is_foreign: Union[Unset, int] = UNSET
    jobs_enabled: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        data_source: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.data_source, Unset):
            data_source = self.data_source.to_dict()

        data_store: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.data_store, Unset):
            data_store = self.data_store.to_dict()

        owning_organization = self.owning_organization
        allowed_teams: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.allowed_teams, Unset):
            allowed_teams = []
            for allowed_teams_item_data in self.allowed_teams:
                allowed_teams_item = allowed_teams_item_data

                allowed_teams.append(allowed_teams_item)

        is_active = self.is_active
        is_public = self.is_public
        description = self.description
        display_name = self.display_name
        created = self.created
        is_editable = self.is_editable
        last_used_by_user = self.last_used_by_user
        logo_url = self.logo_url
        sorting_key = self.sorting_key
        details = self.details
        publication = self.publication
        is_unreported = self.is_unreported
        is_foreign = self.is_foreign
        jobs_enabled = self.jobs_enabled

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if data_source is not UNSET:
            field_dict["dataSource"] = data_source
        if data_store is not UNSET:
            field_dict["dataStore"] = data_store
        if owning_organization is not UNSET:
            field_dict["owningOrganization"] = owning_organization
        if allowed_teams is not UNSET:
            field_dict["allowedTeams"] = allowed_teams
        if is_active is not UNSET:
            field_dict["isActive"] = is_active
        if is_public is not UNSET:
            field_dict["isPublic"] = is_public
        if description is not UNSET:
            field_dict["description"] = description
        if display_name is not UNSET:
            field_dict["displayName"] = display_name
        if created is not UNSET:
            field_dict["created"] = created
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
        if publication is not UNSET:
            field_dict["publication"] = publication
        if is_unreported is not UNSET:
            field_dict["isUnreported"] = is_unreported
        if is_foreign is not UNSET:
            field_dict["isForeign"] = is_foreign
        if jobs_enabled is not UNSET:
            field_dict["jobsEnabled"] = jobs_enabled

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        _data_source = d.pop("dataSource", UNSET)
        data_source: Union[Unset, DatasetInfoResponse200DataSource]
        if isinstance(_data_source, Unset):
            data_source = UNSET
        else:
            data_source = DatasetInfoResponse200DataSource.from_dict(_data_source)

        _data_store = d.pop("dataStore", UNSET)
        data_store: Union[Unset, DatasetInfoResponse200DataStore]
        if isinstance(_data_store, Unset):
            data_store = UNSET
        else:
            data_store = DatasetInfoResponse200DataStore.from_dict(_data_store)

        owning_organization = d.pop("owningOrganization", UNSET)

        allowed_teams = []
        _allowed_teams = d.pop("allowedTeams", UNSET)
        for allowed_teams_item_data in _allowed_teams or []:
            allowed_teams_item = allowed_teams_item_data

            allowed_teams.append(allowed_teams_item)

        is_active = d.pop("isActive", UNSET)

        is_public = d.pop("isPublic", UNSET)

        description = d.pop("description", UNSET)

        display_name = d.pop("displayName", UNSET)

        created = d.pop("created", UNSET)

        is_editable = d.pop("isEditable", UNSET)

        last_used_by_user = d.pop("lastUsedByUser", UNSET)

        logo_url = d.pop("logoUrl", UNSET)

        sorting_key = d.pop("sortingKey", UNSET)

        details = d.pop("details", UNSET)

        publication = d.pop("publication", UNSET)

        is_unreported = d.pop("isUnreported", UNSET)

        is_foreign = d.pop("isForeign", UNSET)

        jobs_enabled = d.pop("jobsEnabled", UNSET)

        dataset_info_response_200 = cls(
            name=name,
            data_source=data_source,
            data_store=data_store,
            owning_organization=owning_organization,
            allowed_teams=allowed_teams,
            is_active=is_active,
            is_public=is_public,
            description=description,
            display_name=display_name,
            created=created,
            is_editable=is_editable,
            last_used_by_user=last_used_by_user,
            logo_url=logo_url,
            sorting_key=sorting_key,
            details=details,
            publication=publication,
            is_unreported=is_unreported,
            is_foreign=is_foreign,
            jobs_enabled=jobs_enabled,
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

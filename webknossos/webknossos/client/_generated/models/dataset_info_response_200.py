from typing import Any, Dict, List, Type, TypeVar

import attr

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
    """ """

    name: str
    data_source: DatasetInfoResponse200DataSource
    data_store: DatasetInfoResponse200DataStore
    owning_organization: str
    allowed_teams: List[DatasetInfoResponse200AllowedTeamsItem]
    is_active: int
    is_public: int
    description: str
    display_name: str
    created: int
    is_editable: int
    last_used_by_user: int
    logo_url: str
    sorting_key: int
    details: str
    publication: str
    is_unreported: int
    is_foreign: int
    jobs_enabled: int
    tags: List[Any]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        data_source = self.data_source.to_dict()

        data_store = self.data_store.to_dict()

        owning_organization = self.owning_organization
        allowed_teams = []
        for allowed_teams_item_data in self.allowed_teams:
            allowed_teams_item = allowed_teams_item_data.to_dict()

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
        tags = []
        for tags_item_data in self.tags:
            tags_item = tags_item_data

            tags.append(tags_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "dataSource": data_source,
                "dataStore": data_store,
                "owningOrganization": owning_organization,
                "allowedTeams": allowed_teams,
                "isActive": is_active,
                "isPublic": is_public,
                "description": description,
                "displayName": display_name,
                "created": created,
                "isEditable": is_editable,
                "lastUsedByUser": last_used_by_user,
                "logoUrl": logo_url,
                "sortingKey": sorting_key,
                "details": details,
                "publication": publication,
                "isUnreported": is_unreported,
                "isForeign": is_foreign,
                "jobsEnabled": jobs_enabled,
                "tags": tags,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        data_source = DatasetInfoResponse200DataSource.from_dict(d.pop("dataSource"))

        data_store = DatasetInfoResponse200DataStore.from_dict(d.pop("dataStore"))

        owning_organization = d.pop("owningOrganization")

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

        is_editable = d.pop("isEditable")

        last_used_by_user = d.pop("lastUsedByUser")

        logo_url = d.pop("logoUrl")

        sorting_key = d.pop("sortingKey")

        details = d.pop("details")

        publication = d.pop("publication")

        is_unreported = d.pop("isUnreported")

        is_foreign = d.pop("isForeign")

        jobs_enabled = d.pop("jobsEnabled")

        tags = []
        _tags = d.pop("tags")
        for tags_item_data in _tags:
            tags_item = tags_item_data

            tags.append(tags_item)

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
            tags=tags,
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

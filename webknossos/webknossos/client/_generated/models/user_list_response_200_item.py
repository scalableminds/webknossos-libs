from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.user_list_response_200_item_experiences import (
    UserListResponse200ItemExperiences,
)
from ..models.user_list_response_200_item_novel_user_experience_infos import (
    UserListResponse200ItemNovelUserExperienceInfos,
)
from ..models.user_list_response_200_item_teams_item import (
    UserListResponse200ItemTeamsItem,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="UserListResponse200Item")


@attr.s(auto_attribs=True)
class UserListResponse200Item:
    """ """

    id: str
    email: str
    first_name: str
    last_name: str
    is_admin: int
    is_dataset_manager: int
    is_active: int
    teams: List[UserListResponse200ItemTeamsItem]
    experiences: UserListResponse200ItemExperiences
    last_activity: int
    is_anonymous: int
    is_editable: int
    organization: str
    selected_theme: str
    created: int
    last_task_type_id: str
    novel_user_experience_infos: Union[
        Unset, UserListResponse200ItemNovelUserExperienceInfos
    ] = UNSET
    is_super_user: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        email = self.email
        first_name = self.first_name
        last_name = self.last_name
        is_admin = self.is_admin
        is_dataset_manager = self.is_dataset_manager
        is_active = self.is_active
        teams = []
        for teams_item_data in self.teams:
            teams_item = teams_item_data.to_dict()

            teams.append(teams_item)

        experiences = self.experiences.to_dict()

        last_activity = self.last_activity
        is_anonymous = self.is_anonymous
        is_editable = self.is_editable
        organization = self.organization
        selected_theme = self.selected_theme
        created = self.created
        last_task_type_id = self.last_task_type_id
        novel_user_experience_infos: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.novel_user_experience_infos, Unset):
            novel_user_experience_infos = self.novel_user_experience_infos.to_dict()

        is_super_user = self.is_super_user

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "email": email,
                "firstName": first_name,
                "lastName": last_name,
                "isAdmin": is_admin,
                "isDatasetManager": is_dataset_manager,
                "isActive": is_active,
                "teams": teams,
                "experiences": experiences,
                "lastActivity": last_activity,
                "isAnonymous": is_anonymous,
                "isEditable": is_editable,
                "organization": organization,
                "selectedTheme": selected_theme,
                "created": created,
                "lastTaskTypeId": last_task_type_id,
            }
        )
        if novel_user_experience_infos is not UNSET:
            field_dict["novelUserExperienceInfos"] = novel_user_experience_infos
        if is_super_user is not UNSET:
            field_dict["isSuperUser"] = is_super_user

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        email = d.pop("email")

        first_name = d.pop("firstName")

        last_name = d.pop("lastName")

        is_admin = d.pop("isAdmin")

        is_dataset_manager = d.pop("isDatasetManager")

        is_active = d.pop("isActive")

        teams = []
        _teams = d.pop("teams")
        for teams_item_data in _teams:
            teams_item = UserListResponse200ItemTeamsItem.from_dict(teams_item_data)

            teams.append(teams_item)

        experiences = UserListResponse200ItemExperiences.from_dict(d.pop("experiences"))

        last_activity = d.pop("lastActivity")

        is_anonymous = d.pop("isAnonymous")

        is_editable = d.pop("isEditable")

        organization = d.pop("organization")

        selected_theme = d.pop("selectedTheme")

        created = d.pop("created")

        last_task_type_id = d.pop("lastTaskTypeId")

        _novel_user_experience_infos = d.pop("novelUserExperienceInfos", UNSET)
        novel_user_experience_infos: Union[
            Unset, UserListResponse200ItemNovelUserExperienceInfos
        ]
        if isinstance(_novel_user_experience_infos, Unset):
            novel_user_experience_infos = UNSET
        else:
            novel_user_experience_infos = (
                UserListResponse200ItemNovelUserExperienceInfos.from_dict(
                    _novel_user_experience_infos
                )
            )

        is_super_user = d.pop("isSuperUser", UNSET)

        user_list_response_200_item = cls(
            id=id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
            is_dataset_manager=is_dataset_manager,
            is_active=is_active,
            teams=teams,
            experiences=experiences,
            last_activity=last_activity,
            is_anonymous=is_anonymous,
            is_editable=is_editable,
            organization=organization,
            selected_theme=selected_theme,
            created=created,
            last_task_type_id=last_task_type_id,
            novel_user_experience_infos=novel_user_experience_infos,
            is_super_user=is_super_user,
        )

        user_list_response_200_item.additional_properties = d
        return user_list_response_200_item

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

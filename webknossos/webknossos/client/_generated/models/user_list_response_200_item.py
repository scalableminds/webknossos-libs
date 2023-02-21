from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_list_response_200_item_experiences import (
        UserListResponse200ItemExperiences,
    )
    from ..models.user_list_response_200_item_novel_user_experience_infos import (
        UserListResponse200ItemNovelUserExperienceInfos,
    )
    from ..models.user_list_response_200_item_teams_item import (
        UserListResponse200ItemTeamsItem,
    )


T = TypeVar("T", bound="UserListResponse200Item")


@attr.s(auto_attribs=True)
class UserListResponse200Item:
    """
    Attributes:
        id (str):
        email (str):
        first_name (str):
        last_name (str):
        is_admin (int):
        is_dataset_manager (int):
        is_active (int):
        experiences (UserListResponse200ItemExperiences):
        last_activity (int):
        organization (str):
        created (int):
        is_organization_owner (Union[Unset, int]):
        teams (Union[Unset, List['UserListResponse200ItemTeamsItem']]):
        is_anonymous (Union[Unset, int]):
        is_editable (Union[Unset, int]):
        novel_user_experience_infos (Union[Unset, UserListResponse200ItemNovelUserExperienceInfos]):
        selected_theme (Union[Unset, str]):
        last_task_type_id (Union[Unset, str]):
        is_super_user (Union[Unset, int]):
    """

    id: str
    email: str
    first_name: str
    last_name: str
    is_admin: int
    is_dataset_manager: int
    is_active: int
    experiences: "UserListResponse200ItemExperiences"
    last_activity: int
    organization: str
    created: int
    is_organization_owner: Union[Unset, int] = UNSET
    teams: Union[Unset, List["UserListResponse200ItemTeamsItem"]] = UNSET
    is_anonymous: Union[Unset, int] = UNSET
    is_editable: Union[Unset, int] = UNSET
    novel_user_experience_infos: Union[
        Unset, "UserListResponse200ItemNovelUserExperienceInfos"
    ] = UNSET
    selected_theme: Union[Unset, str] = UNSET
    last_task_type_id: Union[Unset, str] = UNSET
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
        experiences = self.experiences.to_dict()

        last_activity = self.last_activity
        organization = self.organization
        created = self.created
        is_organization_owner = self.is_organization_owner
        teams: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.teams, Unset):
            teams = []
            for teams_item_data in self.teams:
                teams_item = teams_item_data.to_dict()

                teams.append(teams_item)

        is_anonymous = self.is_anonymous
        is_editable = self.is_editable
        novel_user_experience_infos: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.novel_user_experience_infos, Unset):
            novel_user_experience_infos = self.novel_user_experience_infos.to_dict()

        selected_theme = self.selected_theme
        last_task_type_id = self.last_task_type_id
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
                "experiences": experiences,
                "lastActivity": last_activity,
                "organization": organization,
                "created": created,
            }
        )
        if is_organization_owner is not UNSET:
            field_dict["isOrganizationOwner"] = is_organization_owner
        if teams is not UNSET:
            field_dict["teams"] = teams
        if is_anonymous is not UNSET:
            field_dict["isAnonymous"] = is_anonymous
        if is_editable is not UNSET:
            field_dict["isEditable"] = is_editable
        if novel_user_experience_infos is not UNSET:
            field_dict["novelUserExperienceInfos"] = novel_user_experience_infos
        if selected_theme is not UNSET:
            field_dict["selectedTheme"] = selected_theme
        if last_task_type_id is not UNSET:
            field_dict["lastTaskTypeId"] = last_task_type_id
        if is_super_user is not UNSET:
            field_dict["isSuperUser"] = is_super_user

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.user_list_response_200_item_experiences import (
            UserListResponse200ItemExperiences,
        )
        from ..models.user_list_response_200_item_novel_user_experience_infos import (
            UserListResponse200ItemNovelUserExperienceInfos,
        )
        from ..models.user_list_response_200_item_teams_item import (
            UserListResponse200ItemTeamsItem,
        )

        d = src_dict.copy()
        id = d.pop("id")

        email = d.pop("email")

        first_name = d.pop("firstName")

        last_name = d.pop("lastName")

        is_admin = d.pop("isAdmin")

        is_dataset_manager = d.pop("isDatasetManager")

        is_active = d.pop("isActive")

        experiences = UserListResponse200ItemExperiences.from_dict(d.pop("experiences"))

        last_activity = d.pop("lastActivity")

        organization = d.pop("organization")

        created = d.pop("created")

        is_organization_owner = d.pop("isOrganizationOwner", UNSET)

        teams = []
        _teams = d.pop("teams", UNSET)
        for teams_item_data in _teams or []:
            teams_item = UserListResponse200ItemTeamsItem.from_dict(teams_item_data)

            teams.append(teams_item)

        is_anonymous = d.pop("isAnonymous", UNSET)

        is_editable = d.pop("isEditable", UNSET)

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

        selected_theme = d.pop("selectedTheme", UNSET)

        last_task_type_id = d.pop("lastTaskTypeId", UNSET)

        is_super_user = d.pop("isSuperUser", UNSET)

        user_list_response_200_item = cls(
            id=id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
            is_dataset_manager=is_dataset_manager,
            is_active=is_active,
            experiences=experiences,
            last_activity=last_activity,
            organization=organization,
            created=created,
            is_organization_owner=is_organization_owner,
            teams=teams,
            is_anonymous=is_anonymous,
            is_editable=is_editable,
            novel_user_experience_infos=novel_user_experience_infos,
            selected_theme=selected_theme,
            last_task_type_id=last_task_type_id,
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

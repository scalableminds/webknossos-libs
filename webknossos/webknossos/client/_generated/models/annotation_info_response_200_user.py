from typing import Any, Dict, List, Type, TypeVar

import attr

from ..models.annotation_info_response_200_user_teams_item import (
    AnnotationInfoResponse200UserTeamsItem,
)

T = TypeVar("T", bound="AnnotationInfoResponse200User")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200User:
    """ """

    id: str
    email: str
    first_name: str
    last_name: str
    is_admin: int
    is_dataset_manager: int
    is_anonymous: int
    teams: List[AnnotationInfoResponse200UserTeamsItem]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        email = self.email
        first_name = self.first_name
        last_name = self.last_name
        is_admin = self.is_admin
        is_dataset_manager = self.is_dataset_manager
        is_anonymous = self.is_anonymous
        teams = []
        for teams_item_data in self.teams:
            teams_item = teams_item_data.to_dict()

            teams.append(teams_item)

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
                "isAnonymous": is_anonymous,
                "teams": teams,
            }
        )

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

        is_anonymous = d.pop("isAnonymous")

        teams = []
        _teams = d.pop("teams")
        for teams_item_data in _teams:
            teams_item = AnnotationInfoResponse200UserTeamsItem.from_dict(
                teams_item_data
            )

            teams.append(teams_item)

        annotation_info_response_200_user = cls(
            id=id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
            is_dataset_manager=is_dataset_manager,
            is_anonymous=is_anonymous,
            teams=teams,
        )

        annotation_info_response_200_user.additional_properties = d
        return annotation_info_response_200_user

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

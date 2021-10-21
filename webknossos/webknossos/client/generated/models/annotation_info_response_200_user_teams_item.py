from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="AnnotationInfoResponse200UserTeamsItem")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200UserTeamsItem:
    """ """

    id: str
    name: str
    is_team_manager: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        name = self.name
        is_team_manager = self.is_team_manager

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "isTeamManager": is_team_manager,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        name = d.pop("name")

        is_team_manager = d.pop("isTeamManager")

        annotation_info_response_200_user_teams_item = cls(
            id=id,
            name=name,
            is_team_manager=is_team_manager,
        )

        annotation_info_response_200_user_teams_item.additional_properties = d
        return annotation_info_response_200_user_teams_item

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

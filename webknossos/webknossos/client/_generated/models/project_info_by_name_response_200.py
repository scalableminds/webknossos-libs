from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.project_info_by_name_response_200_owner import (
        ProjectInfoByNameResponse200Owner,
    )


T = TypeVar("T", bound="ProjectInfoByNameResponse200")


@attr.s(auto_attribs=True)
class ProjectInfoByNameResponse200:
    """
    Attributes:
        name (str):
        team (str):
        team_name (str):
        priority (int):
        paused (int):
        expected_time (int):
        id (str):
        created (int):
        owner (Union[Unset, ProjectInfoByNameResponse200Owner]):
        is_blacklisted_from_report (Union[Unset, int]):
    """

    name: str
    team: str
    team_name: str
    priority: int
    paused: int
    expected_time: int
    id: str
    created: int
    owner: Union[Unset, "ProjectInfoByNameResponse200Owner"] = UNSET
    is_blacklisted_from_report: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        team = self.team
        team_name = self.team_name
        priority = self.priority
        paused = self.paused
        expected_time = self.expected_time
        id = self.id
        created = self.created
        owner: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.owner, Unset):
            owner = self.owner.to_dict()

        is_blacklisted_from_report = self.is_blacklisted_from_report

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "team": team,
                "teamName": team_name,
                "priority": priority,
                "paused": paused,
                "expectedTime": expected_time,
                "id": id,
                "created": created,
            }
        )
        if owner is not UNSET:
            field_dict["owner"] = owner
        if is_blacklisted_from_report is not UNSET:
            field_dict["isBlacklistedFromReport"] = is_blacklisted_from_report

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.project_info_by_name_response_200_owner import (
            ProjectInfoByNameResponse200Owner,
        )

        d = src_dict.copy()
        name = d.pop("name")

        team = d.pop("team")

        team_name = d.pop("teamName")

        priority = d.pop("priority")

        paused = d.pop("paused")

        expected_time = d.pop("expectedTime")

        id = d.pop("id")

        created = d.pop("created")

        _owner = d.pop("owner", UNSET)
        owner: Union[Unset, ProjectInfoByNameResponse200Owner]
        if isinstance(_owner, Unset):
            owner = UNSET
        else:
            owner = ProjectInfoByNameResponse200Owner.from_dict(_owner)

        is_blacklisted_from_report = d.pop("isBlacklistedFromReport", UNSET)

        project_info_by_name_response_200 = cls(
            name=name,
            team=team,
            team_name=team_name,
            priority=priority,
            paused=paused,
            expected_time=expected_time,
            id=id,
            created=created,
            owner=owner,
            is_blacklisted_from_report=is_blacklisted_from_report,
        )

        project_info_by_name_response_200.additional_properties = d
        return project_info_by_name_response_200

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

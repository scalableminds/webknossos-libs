from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.task_info_response_200_type_settings import (
    TaskInfoResponse200TypeSettings,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="TaskInfoResponse200Type")


@attr.s(auto_attribs=True)
class TaskInfoResponse200Type:
    """ """

    id: str
    description: str
    team_name: str
    summary: Union[Unset, str] = UNSET
    team_id: Union[Unset, str] = UNSET
    settings: Union[Unset, TaskInfoResponse200TypeSettings] = UNSET
    recommended_configuration: Union[Unset, str] = UNSET
    tracing_type: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        description = self.description
        team_name = self.team_name
        summary = self.summary
        team_id = self.team_id
        settings: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.settings, Unset):
            settings = self.settings.to_dict()

        recommended_configuration = self.recommended_configuration
        tracing_type = self.tracing_type

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "description": description,
                "teamName": team_name,
            }
        )
        if summary is not UNSET:
            field_dict["summary"] = summary
        if team_id is not UNSET:
            field_dict["teamId"] = team_id
        if settings is not UNSET:
            field_dict["settings"] = settings
        if recommended_configuration is not UNSET:
            field_dict["recommendedConfiguration"] = recommended_configuration
        if tracing_type is not UNSET:
            field_dict["tracingType"] = tracing_type

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        description = d.pop("description")

        team_name = d.pop("teamName")

        summary = d.pop("summary", UNSET)

        team_id = d.pop("teamId", UNSET)

        _settings = d.pop("settings", UNSET)
        settings: Union[Unset, TaskInfoResponse200TypeSettings]
        if isinstance(_settings, Unset):
            settings = UNSET
        else:
            settings = TaskInfoResponse200TypeSettings.from_dict(_settings)

        recommended_configuration = d.pop("recommendedConfiguration", UNSET)

        tracing_type = d.pop("tracingType", UNSET)

        task_info_response_200_type = cls(
            id=id,
            description=description,
            team_name=team_name,
            summary=summary,
            team_id=team_id,
            settings=settings,
            recommended_configuration=recommended_configuration,
            tracing_type=tracing_type,
        )

        task_info_response_200_type.additional_properties = d
        return task_info_response_200_type

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

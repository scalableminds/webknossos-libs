from typing import Any, Dict, List, Optional, Type, TypeVar, cast

import attr

from ..models.task_info_response_200_needed_experience import (
    TaskInfoResponse200NeededExperience,
)
from ..models.task_info_response_200_status import TaskInfoResponse200Status
from ..models.task_info_response_200_type import TaskInfoResponse200Type

T = TypeVar("T", bound="TaskInfoResponse200")


@attr.s(auto_attribs=True)
class TaskInfoResponse200:
    """ """

    id: str
    formatted_hash: str
    project_id: str
    project_name: str
    team: str
    type: TaskInfoResponse200Type
    data_set: str
    needed_experience: TaskInfoResponse200NeededExperience
    created: int
    status: TaskInfoResponse200Status
    script: str
    creation_info: str
    bounding_box: str
    edit_position: List[int]
    edit_rotation: List[int]
    tracing_time: Optional[int]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        formatted_hash = self.formatted_hash
        project_id = self.project_id
        project_name = self.project_name
        team = self.team
        type = self.type.to_dict()

        data_set = self.data_set
        needed_experience = self.needed_experience.to_dict()

        created = self.created
        status = self.status.to_dict()

        script = self.script
        creation_info = self.creation_info
        bounding_box = self.bounding_box
        edit_position = self.edit_position

        edit_rotation = self.edit_rotation

        tracing_time = self.tracing_time

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "formattedHash": formatted_hash,
                "projectId": project_id,
                "projectName": project_name,
                "team": team,
                "type": type,
                "dataSet": data_set,
                "neededExperience": needed_experience,
                "created": created,
                "status": status,
                "script": script,
                "creationInfo": creation_info,
                "boundingBox": bounding_box,
                "editPosition": edit_position,
                "editRotation": edit_rotation,
                "tracingTime": tracing_time,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        formatted_hash = d.pop("formattedHash")

        project_id = d.pop("projectId")

        project_name = d.pop("projectName")

        team = d.pop("team")

        type = TaskInfoResponse200Type.from_dict(d.pop("type"))

        data_set = d.pop("dataSet")

        needed_experience = TaskInfoResponse200NeededExperience.from_dict(
            d.pop("neededExperience")
        )

        created = d.pop("created")

        status = TaskInfoResponse200Status.from_dict(d.pop("status"))

        script = d.pop("script")

        creation_info = d.pop("creationInfo")

        bounding_box = d.pop("boundingBox")

        edit_position = cast(List[int], d.pop("editPosition"))

        edit_rotation = cast(List[int], d.pop("editRotation"))

        tracing_time = d.pop("tracingTime")

        task_info_response_200 = cls(
            id=id,
            formatted_hash=formatted_hash,
            project_id=project_id,
            project_name=project_name,
            team=team,
            type=type,
            data_set=data_set,
            needed_experience=needed_experience,
            created=created,
            status=status,
            script=script,
            creation_info=creation_info,
            bounding_box=bounding_box,
            edit_position=edit_position,
            edit_rotation=edit_rotation,
            tracing_time=tracing_time,
        )

        task_info_response_200.additional_properties = d
        return task_info_response_200

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

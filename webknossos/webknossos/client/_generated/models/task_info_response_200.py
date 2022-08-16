from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..models.task_info_response_200_needed_experience import (
    TaskInfoResponse200NeededExperience,
)
from ..models.task_info_response_200_status import TaskInfoResponse200Status
from ..models.task_info_response_200_type import TaskInfoResponse200Type
from ..types import UNSET, Unset

T = TypeVar("T", bound="TaskInfoResponse200")


@attr.s(auto_attribs=True)
class TaskInfoResponse200:
    """ """

    id: str
    project_id: str
    team: str
    data_set: str
    created: int
    status: TaskInfoResponse200Status
    bounding_box: str
    formatted_hash: Union[Unset, str] = UNSET
    project_name: Union[Unset, str] = UNSET
    type: Union[Unset, TaskInfoResponse200Type] = UNSET
    needed_experience: Union[Unset, TaskInfoResponse200NeededExperience] = UNSET
    script: Union[Unset, str] = UNSET
    tracing_time: Union[Unset, None, int] = UNSET
    creation_info: Union[Unset, str] = UNSET
    edit_position: Union[Unset, List[int]] = UNSET
    edit_rotation: Union[Unset, List[int]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        project_id = self.project_id
        team = self.team
        data_set = self.data_set
        created = self.created
        status = self.status.to_dict()

        bounding_box = self.bounding_box
        formatted_hash = self.formatted_hash
        project_name = self.project_name
        type: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.type, Unset):
            type = self.type.to_dict()

        needed_experience: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.needed_experience, Unset):
            needed_experience = self.needed_experience.to_dict()

        script = self.script
        tracing_time = self.tracing_time
        creation_info = self.creation_info
        edit_position: Union[Unset, List[int]] = UNSET
        if not isinstance(self.edit_position, Unset):
            edit_position = self.edit_position

        edit_rotation: Union[Unset, List[int]] = UNSET
        if not isinstance(self.edit_rotation, Unset):
            edit_rotation = self.edit_rotation

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "projectId": project_id,
                "team": team,
                "dataSet": data_set,
                "created": created,
                "status": status,
                "boundingBox": bounding_box,
            }
        )
        if formatted_hash is not UNSET:
            field_dict["formattedHash"] = formatted_hash
        if project_name is not UNSET:
            field_dict["projectName"] = project_name
        if type is not UNSET:
            field_dict["type"] = type
        if needed_experience is not UNSET:
            field_dict["neededExperience"] = needed_experience
        if script is not UNSET:
            field_dict["script"] = script
        if tracing_time is not UNSET:
            field_dict["tracingTime"] = tracing_time
        if creation_info is not UNSET:
            field_dict["creationInfo"] = creation_info
        if edit_position is not UNSET:
            field_dict["editPosition"] = edit_position
        if edit_rotation is not UNSET:
            field_dict["editRotation"] = edit_rotation

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        project_id = d.pop("projectId")

        team = d.pop("team")

        data_set = d.pop("dataSet")

        created = d.pop("created")

        status = TaskInfoResponse200Status.from_dict(d.pop("status"))

        bounding_box = d.pop("boundingBox")

        formatted_hash = d.pop("formattedHash", UNSET)

        project_name = d.pop("projectName", UNSET)

        _type = d.pop("type", UNSET)
        type: Union[Unset, TaskInfoResponse200Type]
        if isinstance(_type, Unset):
            type = UNSET
        else:
            type = TaskInfoResponse200Type.from_dict(_type)

        _needed_experience = d.pop("neededExperience", UNSET)
        needed_experience: Union[Unset, TaskInfoResponse200NeededExperience]
        if isinstance(_needed_experience, Unset):
            needed_experience = UNSET
        else:
            needed_experience = TaskInfoResponse200NeededExperience.from_dict(
                _needed_experience
            )

        script = d.pop("script", UNSET)

        tracing_time = d.pop("tracingTime", UNSET)

        creation_info = d.pop("creationInfo", UNSET)

        edit_position = cast(List[int], d.pop("editPosition", UNSET))

        edit_rotation = cast(List[int], d.pop("editRotation", UNSET))

        task_info_response_200 = cls(
            id=id,
            project_id=project_id,
            team=team,
            data_set=data_set,
            created=created,
            status=status,
            bounding_box=bounding_box,
            formatted_hash=formatted_hash,
            project_name=project_name,
            type=type,
            needed_experience=needed_experience,
            script=script,
            tracing_time=tracing_time,
            creation_info=creation_info,
            edit_position=edit_position,
            edit_rotation=edit_rotation,
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

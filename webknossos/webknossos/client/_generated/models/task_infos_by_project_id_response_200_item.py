from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_infos_by_project_id_response_200_item_needed_experience import (
        TaskInfosByProjectIdResponse200ItemNeededExperience,
    )
    from ..models.task_infos_by_project_id_response_200_item_status import (
        TaskInfosByProjectIdResponse200ItemStatus,
    )
    from ..models.task_infos_by_project_id_response_200_item_type import (
        TaskInfosByProjectIdResponse200ItemType,
    )


T = TypeVar("T", bound="TaskInfosByProjectIdResponse200Item")


@attr.s(auto_attribs=True)
class TaskInfosByProjectIdResponse200Item:
    """
    Attributes:
        id (str):
        project_id (str):
        team (str):
        data_set (str):
        created (int):
        status (TaskInfosByProjectIdResponse200ItemStatus):
        bounding_box (str):
        formatted_hash (Union[Unset, str]):
        project_name (Union[Unset, str]):
        type (Union[Unset, TaskInfosByProjectIdResponse200ItemType]):
        needed_experience (Union[Unset, TaskInfosByProjectIdResponse200ItemNeededExperience]):
        script (Union[Unset, str]):
        tracing_time (Union[Unset, None, int]):
        creation_info (Union[Unset, str]):
        edit_position (Union[Unset, List[int]]):
        edit_rotation (Union[Unset, List[int]]):
    """

    id: str
    project_id: str
    team: str
    data_set: str
    created: int
    status: "TaskInfosByProjectIdResponse200ItemStatus"
    bounding_box: str
    formatted_hash: Union[Unset, str] = UNSET
    project_name: Union[Unset, str] = UNSET
    type: Union[Unset, "TaskInfosByProjectIdResponse200ItemType"] = UNSET
    needed_experience: Union[
        Unset, "TaskInfosByProjectIdResponse200ItemNeededExperience"
    ] = UNSET
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
        from ..models.task_infos_by_project_id_response_200_item_needed_experience import (
            TaskInfosByProjectIdResponse200ItemNeededExperience,
        )
        from ..models.task_infos_by_project_id_response_200_item_status import (
            TaskInfosByProjectIdResponse200ItemStatus,
        )
        from ..models.task_infos_by_project_id_response_200_item_type import (
            TaskInfosByProjectIdResponse200ItemType,
        )

        d = src_dict.copy()
        id = d.pop("id")

        project_id = d.pop("projectId")

        team = d.pop("team")

        data_set = d.pop("dataSet")

        created = d.pop("created")

        status = TaskInfosByProjectIdResponse200ItemStatus.from_dict(d.pop("status"))

        bounding_box = d.pop("boundingBox")

        formatted_hash = d.pop("formattedHash", UNSET)

        project_name = d.pop("projectName", UNSET)

        _type = d.pop("type", UNSET)
        type: Union[Unset, TaskInfosByProjectIdResponse200ItemType]
        if isinstance(_type, Unset):
            type = UNSET
        else:
            type = TaskInfosByProjectIdResponse200ItemType.from_dict(_type)

        _needed_experience = d.pop("neededExperience", UNSET)
        needed_experience: Union[
            Unset, TaskInfosByProjectIdResponse200ItemNeededExperience
        ]
        if isinstance(_needed_experience, Unset):
            needed_experience = UNSET
        else:
            needed_experience = (
                TaskInfosByProjectIdResponse200ItemNeededExperience.from_dict(
                    _needed_experience
                )
            )

        script = d.pop("script", UNSET)

        tracing_time = d.pop("tracingTime", UNSET)

        creation_info = d.pop("creationInfo", UNSET)

        edit_position = cast(List[int], d.pop("editPosition", UNSET))

        edit_rotation = cast(List[int], d.pop("editRotation", UNSET))

        task_infos_by_project_id_response_200_item = cls(
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

        task_infos_by_project_id_response_200_item.additional_properties = d
        return task_infos_by_project_id_response_200_item

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

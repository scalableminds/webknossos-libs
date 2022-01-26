from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="TaskInfoResponse200Status")


@attr.s(auto_attribs=True)
class TaskInfoResponse200Status:
    """ """

    open_: int
    active: int
    finished: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        open_ = self.open_
        active = self.active
        finished = self.finished

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "open": open_,
                "active": active,
                "finished": finished,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        open_ = d.pop("open")

        active = d.pop("active")

        finished = d.pop("finished")

        task_info_response_200_status = cls(
            open_=open_,
            active=active,
            finished=finished,
        )

        task_info_response_200_status.additional_properties = d
        return task_info_response_200_status

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

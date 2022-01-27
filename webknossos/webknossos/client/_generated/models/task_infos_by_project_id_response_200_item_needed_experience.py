from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="TaskInfosByProjectIdResponse200ItemNeededExperience")


@attr.s(auto_attribs=True)
class TaskInfosByProjectIdResponse200ItemNeededExperience:
    """ """

    domain: str
    value: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        domain = self.domain
        value = self.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "domain": domain,
                "value": value,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        domain = d.pop("domain")

        value = d.pop("value")

        task_infos_by_project_id_response_200_item_needed_experience = cls(
            domain=domain,
            value=value,
        )

        task_infos_by_project_id_response_200_item_needed_experience.additional_properties = (
            d
        )
        return task_infos_by_project_id_response_200_item_needed_experience

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

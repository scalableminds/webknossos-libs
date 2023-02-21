from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="TaskInfosByProjectIdResponse200ItemNeededExperience")


@attr.s(auto_attribs=True)
class TaskInfosByProjectIdResponse200ItemNeededExperience:
    """
    Attributes:
        domain (Union[Unset, str]):
        value (Union[Unset, int]):
    """

    domain: Union[Unset, str] = UNSET
    value: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        domain = self.domain
        value = self.value

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if domain is not UNSET:
            field_dict["domain"] = domain
        if value is not UNSET:
            field_dict["value"] = value

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        domain = d.pop("domain", UNSET)

        value = d.pop("value", UNSET)

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

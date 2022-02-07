from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="UserListResponse200ItemExperiences")


@attr.s(auto_attribs=True)
class UserListResponse200ItemExperiences:
    """ """

    sample_exp: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        sample_exp = self.sample_exp

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if sample_exp is not UNSET:
            field_dict["sampleExp"] = sample_exp

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        sample_exp = d.pop("sampleExp", UNSET)

        user_list_response_200_item_experiences = cls(
            sample_exp=sample_exp,
        )

        user_list_response_200_item_experiences.additional_properties = d
        return user_list_response_200_item_experiences

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

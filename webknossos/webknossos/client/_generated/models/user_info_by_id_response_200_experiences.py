from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="UserInfoByIdResponse200Experiences")


@attr.s(auto_attribs=True)
class UserInfoByIdResponse200Experiences:
    """ """

    sample_exp: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        sample_exp = self.sample_exp

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "sampleExp": sample_exp,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        sample_exp = d.pop("sampleExp")

        user_info_by_id_response_200_experiences = cls(
            sample_exp=sample_exp,
        )

        user_info_by_id_response_200_experiences.additional_properties = d
        return user_info_by_id_response_200_experiences

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

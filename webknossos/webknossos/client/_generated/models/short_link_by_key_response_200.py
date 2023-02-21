from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="ShortLinkByKeyResponse200")


@attr.s(auto_attribs=True)
class ShortLinkByKeyResponse200:
    """
    Attributes:
        long_link (str):
        id (Union[Unset, str]):
        key (Union[Unset, str]):
    """

    long_link: str
    id: Union[Unset, str] = UNSET
    key: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        long_link = self.long_link
        id = self.id
        key = self.key

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "longLink": long_link,
            }
        )
        if id is not UNSET:
            field_dict["_id"] = id
        if key is not UNSET:
            field_dict["key"] = key

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        long_link = d.pop("longLink")

        id = d.pop("_id", UNSET)

        key = d.pop("key", UNSET)

        short_link_by_key_response_200 = cls(
            long_link=long_link,
            id=id,
            key=key,
        )

        short_link_by_key_response_200.additional_properties = d
        return short_link_by_key_response_200

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

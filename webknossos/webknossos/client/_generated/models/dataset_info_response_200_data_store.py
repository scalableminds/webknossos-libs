from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetInfoResponse200DataStore")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataStore:
    """
    Attributes:
        name (str):
        url (str):
        allows_upload (int):
        is_scratch (Union[Unset, int]):
    """

    name: str
    url: str
    allows_upload: int
    is_scratch: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        url = self.url
        allows_upload = self.allows_upload
        is_scratch = self.is_scratch

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "url": url,
                "allowsUpload": allows_upload,
            }
        )
        if is_scratch is not UNSET:
            field_dict["isScratch"] = is_scratch

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        url = d.pop("url")

        allows_upload = d.pop("allowsUpload")

        is_scratch = d.pop("isScratch", UNSET)

        dataset_info_response_200_data_store = cls(
            name=name,
            url=url,
            allows_upload=allows_upload,
            is_scratch=is_scratch,
        )

        dataset_info_response_200_data_store.additional_properties = d
        return dataset_info_response_200_data_store

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

from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="DatasetInfoResponse200DataStore")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataStore:
    """ """

    name: str
    url: str
    is_foreign: int
    is_scratch: int
    is_connector: int
    allows_upload: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        url = self.url
        is_foreign = self.is_foreign
        is_scratch = self.is_scratch
        is_connector = self.is_connector
        allows_upload = self.allows_upload

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "url": url,
                "isForeign": is_foreign,
                "isScratch": is_scratch,
                "isConnector": is_connector,
                "allowsUpload": allows_upload,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        url = d.pop("url")

        is_foreign = d.pop("isForeign")

        is_scratch = d.pop("isScratch")

        is_connector = d.pop("isConnector")

        allows_upload = d.pop("allowsUpload")

        dataset_info_response_200_data_store = cls(
            name=name,
            url=url,
            is_foreign=is_foreign,
            is_scratch=is_scratch,
            is_connector=is_connector,
            allows_upload=allows_upload,
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

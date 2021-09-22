from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfoResponse200DataStore")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200DataStore:
    """ """

    name: Union[Unset, str] = UNSET
    url: Union[Unset, str] = UNSET
    is_foreign: Union[Unset, int] = UNSET
    is_scratch: Union[Unset, int] = UNSET
    is_connector: Union[Unset, int] = UNSET
    allows_upload: Union[Unset, int] = UNSET
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
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if url is not UNSET:
            field_dict["url"] = url
        if is_foreign is not UNSET:
            field_dict["isForeign"] = is_foreign
        if is_scratch is not UNSET:
            field_dict["isScratch"] = is_scratch
        if is_connector is not UNSET:
            field_dict["isConnector"] = is_connector
        if allows_upload is not UNSET:
            field_dict["allowsUpload"] = allows_upload

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        url = d.pop("url", UNSET)

        is_foreign = d.pop("isForeign", UNSET)

        is_scratch = d.pop("isScratch", UNSET)

        is_connector = d.pop("isConnector", UNSET)

        allows_upload = d.pop("allowsUpload", UNSET)

        annotation_info_response_200_data_store = cls(
            name=name,
            url=url,
            is_foreign=is_foreign,
            is_scratch=is_scratch,
            is_connector=is_connector,
            allows_upload=allows_upload,
        )

        annotation_info_response_200_data_store.additional_properties = d
        return annotation_info_response_200_data_store

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

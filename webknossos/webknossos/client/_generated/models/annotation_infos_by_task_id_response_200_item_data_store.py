from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200ItemDataStore")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200ItemDataStore:
    """ """

    name: str
    url: str
    allows_upload: int
    is_scratch: Union[Unset, int] = UNSET
    is_connector: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        url = self.url
        allows_upload = self.allows_upload
        is_scratch = self.is_scratch
        is_connector = self.is_connector

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
        if is_connector is not UNSET:
            field_dict["isConnector"] = is_connector

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        url = d.pop("url")

        allows_upload = d.pop("allowsUpload")

        is_scratch = d.pop("isScratch", UNSET)

        is_connector = d.pop("isConnector", UNSET)

        annotation_infos_by_task_id_response_200_item_data_store = cls(
            name=name,
            url=url,
            allows_upload=allows_upload,
            is_scratch=is_scratch,
            is_connector=is_connector,
        )

        annotation_infos_by_task_id_response_200_item_data_store.additional_properties = (
            d
        )
        return annotation_infos_by_task_id_response_200_item_data_store

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

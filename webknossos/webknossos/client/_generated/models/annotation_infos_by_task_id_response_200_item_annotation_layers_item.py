from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem:
    """
    Attributes:
        typ (str):
        tracing_id (Union[Unset, str]):
        name (Union[Unset, str]):
    """

    typ: str
    tracing_id: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        typ = self.typ
        tracing_id = self.tracing_id
        name = self.name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "typ": typ,
            }
        )
        if tracing_id is not UNSET:
            field_dict["tracingId"] = tracing_id
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        typ = d.pop("typ")

        tracing_id = d.pop("tracingId", UNSET)

        name = d.pop("name", UNSET)

        annotation_infos_by_task_id_response_200_item_annotation_layers_item = cls(
            typ=typ,
            tracing_id=tracing_id,
            name=name,
        )

        annotation_infos_by_task_id_response_200_item_annotation_layers_item.additional_properties = (
            d
        )
        return annotation_infos_by_task_id_response_200_item_annotation_layers_item

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

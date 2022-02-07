from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200ItemAnnotationLayersItem:
    """ """

    tracing_id: str
    typ: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        tracing_id = self.tracing_id
        typ = self.typ

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "tracingId": tracing_id,
                "typ": typ,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        tracing_id = d.pop("tracingId")

        typ = d.pop("typ")

        annotation_infos_by_task_id_response_200_item_annotation_layers_item = cls(
            tracing_id=tracing_id,
            typ=typ,
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

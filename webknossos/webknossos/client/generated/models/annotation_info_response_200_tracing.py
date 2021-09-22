from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfoResponse200Tracing")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200Tracing:
    """ """

    skeleton: Union[Unset, str] = UNSET
    volume: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        skeleton = self.skeleton
        volume = self.volume

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if skeleton is not UNSET:
            field_dict["skeleton"] = skeleton
        if volume is not UNSET:
            field_dict["volume"] = volume

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        skeleton = d.pop("skeleton", UNSET)

        volume = d.pop("volume", UNSET)

        annotation_info_response_200_tracing = cls(
            skeleton=skeleton,
            volume=volume,
        )

        annotation_info_response_200_tracing.additional_properties = d
        return annotation_info_response_200_tracing

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

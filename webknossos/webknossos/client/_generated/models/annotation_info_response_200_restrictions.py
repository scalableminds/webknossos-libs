from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfoResponse200Restrictions")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200Restrictions:
    """ """

    allow_access: Union[Unset, int] = UNSET
    allow_update: Union[Unset, int] = UNSET
    allow_finish: Union[Unset, int] = UNSET
    allow_download: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        allow_access = self.allow_access
        allow_update = self.allow_update
        allow_finish = self.allow_finish
        allow_download = self.allow_download

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allow_access is not UNSET:
            field_dict["allowAccess"] = allow_access
        if allow_update is not UNSET:
            field_dict["allowUpdate"] = allow_update
        if allow_finish is not UNSET:
            field_dict["allowFinish"] = allow_finish
        if allow_download is not UNSET:
            field_dict["allowDownload"] = allow_download

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        allow_access = d.pop("allowAccess", UNSET)

        allow_update = d.pop("allowUpdate", UNSET)

        allow_finish = d.pop("allowFinish", UNSET)

        allow_download = d.pop("allowDownload", UNSET)

        annotation_info_response_200_restrictions = cls(
            allow_access=allow_access,
            allow_update=allow_update,
            allow_finish=allow_finish,
            allow_download=allow_download,
        )

        annotation_info_response_200_restrictions.additional_properties = d
        return annotation_info_response_200_restrictions

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

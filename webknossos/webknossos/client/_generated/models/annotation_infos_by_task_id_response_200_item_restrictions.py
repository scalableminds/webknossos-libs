from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="AnnotationInfosByTaskIdResponse200ItemRestrictions")


@attr.s(auto_attribs=True)
class AnnotationInfosByTaskIdResponse200ItemRestrictions:
    """ """

    allow_access: int
    allow_update: int
    allow_finish: int
    allow_download: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        allow_access = self.allow_access
        allow_update = self.allow_update
        allow_finish = self.allow_finish
        allow_download = self.allow_download

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "allowAccess": allow_access,
                "allowUpdate": allow_update,
                "allowFinish": allow_finish,
                "allowDownload": allow_download,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        allow_access = d.pop("allowAccess")

        allow_update = d.pop("allowUpdate")

        allow_finish = d.pop("allowFinish")

        allow_download = d.pop("allowDownload")

        annotation_infos_by_task_id_response_200_item_restrictions = cls(
            allow_access=allow_access,
            allow_update=allow_update,
            allow_finish=allow_finish,
            allow_download=allow_download,
        )

        annotation_infos_by_task_id_response_200_item_restrictions.additional_properties = (
            d
        )
        return annotation_infos_by_task_id_response_200_item_restrictions

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

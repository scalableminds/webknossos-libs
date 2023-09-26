from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.upload_information_needs_conversion import (
        UploadInformationNeedsConversion,
    )


T = TypeVar("T", bound="UploadInformation")


@attr.s(auto_attribs=True)
class UploadInformation:
    """
    Attributes:
        upload_id (str):
        needs_conversion (Union[Unset, UploadInformationNeedsConversion]):
    """

    upload_id: str
    needs_conversion: Union[Unset, "UploadInformationNeedsConversion"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        upload_id = self.upload_id
        needs_conversion: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.needs_conversion, Unset):
            needs_conversion = self.needs_conversion.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "uploadId": upload_id,
            }
        )
        if needs_conversion is not UNSET:
            field_dict["needsConversion"] = needs_conversion

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.upload_information_needs_conversion import (
            UploadInformationNeedsConversion,
        )

        d = src_dict.copy()
        upload_id = d.pop("uploadId")

        _needs_conversion = d.pop("needsConversion", UNSET)
        needs_conversion: Union[Unset, UploadInformationNeedsConversion]
        if isinstance(_needs_conversion, Unset):
            needs_conversion = UNSET
        else:
            needs_conversion = UploadInformationNeedsConversion.from_dict(
                _needs_conversion
            )

        upload_information = cls(
            upload_id=upload_id,
            needs_conversion=needs_conversion,
        )

        upload_information.additional_properties = d
        return upload_information

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

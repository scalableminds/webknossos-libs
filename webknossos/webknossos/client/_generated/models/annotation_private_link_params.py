from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.instant import Instant


T = TypeVar("T", bound="AnnotationPrivateLinkParams")


@attr.s(auto_attribs=True)
class AnnotationPrivateLinkParams:
    """
    Attributes:
        annotation (str):
        expiration_date_time (Union[Unset, Instant]):
    """

    annotation: str
    expiration_date_time: Union[Unset, "Instant"] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        annotation = self.annotation
        expiration_date_time: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.expiration_date_time, Unset):
            expiration_date_time = self.expiration_date_time.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "annotation": annotation,
            }
        )
        if expiration_date_time is not UNSET:
            field_dict["expirationDateTime"] = expiration_date_time

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.instant import Instant

        d = src_dict.copy()
        annotation = d.pop("annotation")

        _expiration_date_time = d.pop("expirationDateTime", UNSET)
        expiration_date_time: Union[Unset, Instant]
        if isinstance(_expiration_date_time, Unset):
            expiration_date_time = UNSET
        else:
            expiration_date_time = Instant.from_dict(_expiration_date_time)

        annotation_private_link_params = cls(
            annotation=annotation,
            expiration_date_time=expiration_date_time,
        )

        annotation_private_link_params.additional_properties = d
        return annotation_private_link_params

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

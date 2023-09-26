from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="Instant")


@attr.s(auto_attribs=True)
class Instant:
    """
    Attributes:
        epoch_millis (int):
        past (bool):
    """

    epoch_millis: int
    past: bool
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        epoch_millis = self.epoch_millis
        past = self.past

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "epochMillis": epoch_millis,
                "past": past,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        epoch_millis = d.pop("epochMillis")

        past = d.pop("past")

        instant = cls(
            epoch_millis=epoch_millis,
            past=past,
        )

        instant.additional_properties = d
        return instant

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

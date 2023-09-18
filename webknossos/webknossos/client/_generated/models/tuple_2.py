from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr

if TYPE_CHECKING:
    from ..models.tuple_21 import Tuple21
    from ..models.tuple_22 import Tuple22


T = TypeVar("T", bound="Tuple2")


@attr.s(auto_attribs=True)
class Tuple2:
    """
    Attributes:
        field_1 (Tuple21):
        field_2 (Tuple22):
    """

    field_1: "Tuple21"
    field_2: "Tuple22"
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        field_1 = self.field_1.to_dict()

        field_2 = self.field_2.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "_1": field_1,
                "_2": field_2,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.tuple_21 import Tuple21
        from ..models.tuple_22 import Tuple22

        d = src_dict.copy()
        field_1 = Tuple21.from_dict(d.pop("_1"))

        field_2 = Tuple22.from_dict(d.pop("_2"))

        tuple_2 = cls(
            field_1=field_1,
            field_2=field_2,
        )

        tuple_2.additional_properties = d
        return tuple_2

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

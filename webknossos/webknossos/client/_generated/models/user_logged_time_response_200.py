from typing import Any, Dict, List, Type, TypeVar

import attr

from ..models.user_logged_time_response_200_logged_time_item import (
    UserLoggedTimeResponse200LoggedTimeItem,
)

T = TypeVar("T", bound="UserLoggedTimeResponse200")


@attr.s(auto_attribs=True)
class UserLoggedTimeResponse200:
    """ """

    logged_time: List[UserLoggedTimeResponse200LoggedTimeItem]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        logged_time = []
        for logged_time_item_data in self.logged_time:
            logged_time_item = logged_time_item_data.to_dict()

            logged_time.append(logged_time_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "loggedTime": logged_time,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        logged_time = []
        _logged_time = d.pop("loggedTime")
        for logged_time_item_data in _logged_time:
            logged_time_item = UserLoggedTimeResponse200LoggedTimeItem.from_dict(
                logged_time_item_data
            )

            logged_time.append(logged_time_item)

        user_logged_time_response_200 = cls(
            logged_time=logged_time,
        )

        user_logged_time_response_200.additional_properties = d
        return user_logged_time_response_200

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

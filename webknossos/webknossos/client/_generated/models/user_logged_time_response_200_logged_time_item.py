from typing import Any, Dict, List, Type, TypeVar

import attr

from ..models.user_logged_time_response_200_logged_time_item_payment_interval import (
    UserLoggedTimeResponse200LoggedTimeItemPaymentInterval,
)

T = TypeVar("T", bound="UserLoggedTimeResponse200LoggedTimeItem")


@attr.s(auto_attribs=True)
class UserLoggedTimeResponse200LoggedTimeItem:
    """ """

    payment_interval: UserLoggedTimeResponse200LoggedTimeItemPaymentInterval
    duration_in_seconds: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payment_interval = self.payment_interval.to_dict()

        duration_in_seconds = self.duration_in_seconds

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "paymentInterval": payment_interval,
                "durationInSeconds": duration_in_seconds,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        payment_interval = (
            UserLoggedTimeResponse200LoggedTimeItemPaymentInterval.from_dict(
                d.pop("paymentInterval")
            )
        )

        duration_in_seconds = d.pop("durationInSeconds")

        user_logged_time_response_200_logged_time_item = cls(
            payment_interval=payment_interval,
            duration_in_seconds=duration_in_seconds,
        )

        user_logged_time_response_200_logged_time_item.additional_properties = d
        return user_logged_time_response_200_logged_time_item

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

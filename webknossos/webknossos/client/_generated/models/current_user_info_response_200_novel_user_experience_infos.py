from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="CurrentUserInfoResponse200NovelUserExperienceInfos")


@attr.s(auto_attribs=True)
class CurrentUserInfoResponse200NovelUserExperienceInfos:
    """ """

    last_viewed_whats_new_timestamp: Union[Unset, int] = UNSET
    should_see_modern_controls_modal: Union[Unset, int] = UNSET
    has_seen_dashboard_welcome_banner: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        last_viewed_whats_new_timestamp = self.last_viewed_whats_new_timestamp
        should_see_modern_controls_modal = self.should_see_modern_controls_modal
        has_seen_dashboard_welcome_banner = self.has_seen_dashboard_welcome_banner

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if last_viewed_whats_new_timestamp is not UNSET:
            field_dict["lastViewedWhatsNewTimestamp"] = last_viewed_whats_new_timestamp
        if should_see_modern_controls_modal is not UNSET:
            field_dict[
                "shouldSeeModernControlsModal"
            ] = should_see_modern_controls_modal
        if has_seen_dashboard_welcome_banner is not UNSET:
            field_dict[
                "hasSeenDashboardWelcomeBanner"
            ] = has_seen_dashboard_welcome_banner

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        last_viewed_whats_new_timestamp = d.pop("lastViewedWhatsNewTimestamp", UNSET)

        should_see_modern_controls_modal = d.pop("shouldSeeModernControlsModal", UNSET)

        has_seen_dashboard_welcome_banner = d.pop(
            "hasSeenDashboardWelcomeBanner", UNSET
        )

        current_user_info_response_200_novel_user_experience_infos = cls(
            last_viewed_whats_new_timestamp=last_viewed_whats_new_timestamp,
            should_see_modern_controls_modal=should_see_modern_controls_modal,
            has_seen_dashboard_welcome_banner=has_seen_dashboard_welcome_banner,
        )

        current_user_info_response_200_novel_user_experience_infos.additional_properties = (
            d
        )
        return current_user_info_response_200_novel_user_experience_infos

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

from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

T = TypeVar(
    "T", bound="DatasetInfoResponse200DataSourceDataLayersItemAdminViewConfiguration"
)


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSourceDataLayersItemAdminViewConfiguration:
    """ """

    intensity_range: Union[Unset, List[int]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        intensity_range: Union[Unset, List[int]] = UNSET
        if not isinstance(self.intensity_range, Unset):
            intensity_range = self.intensity_range

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if intensity_range is not UNSET:
            field_dict["intensityRange"] = intensity_range

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        intensity_range = cast(List[int], d.pop("intensityRange", UNSET))

        dataset_info_response_200_data_source_data_layers_item_admin_view_configuration = cls(
            intensity_range=intensity_range,
        )

        dataset_info_response_200_data_source_data_layers_item_admin_view_configuration.additional_properties = (
            d
        )
        return dataset_info_response_200_data_source_data_layers_item_admin_view_configuration

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

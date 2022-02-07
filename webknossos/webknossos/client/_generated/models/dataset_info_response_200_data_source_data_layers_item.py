from typing import Any, Dict, List, Type, TypeVar, cast

import attr

from ..models.dataset_info_response_200_data_source_data_layers_item_bounding_box import (
    DatasetInfoResponse200DataSourceDataLayersItemBoundingBox,
)

T = TypeVar("T", bound="DatasetInfoResponse200DataSourceDataLayersItem")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSourceDataLayersItem:
    """ """

    name: str
    category: str
    bounding_box: DatasetInfoResponse200DataSourceDataLayersItemBoundingBox
    resolutions: List[List[int]]
    element_class: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        category = self.category
        bounding_box = self.bounding_box.to_dict()

        resolutions = []
        for resolutions_item_data in self.resolutions:
            resolutions_item = resolutions_item_data

            resolutions.append(resolutions_item)

        element_class = self.element_class

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "category": category,
                "boundingBox": bounding_box,
                "resolutions": resolutions,
                "elementClass": element_class,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        category = d.pop("category")

        bounding_box = (
            DatasetInfoResponse200DataSourceDataLayersItemBoundingBox.from_dict(
                d.pop("boundingBox")
            )
        )

        resolutions = []
        _resolutions = d.pop("resolutions")
        for resolutions_item_data in _resolutions:
            resolutions_item = cast(List[int], resolutions_item_data)

            resolutions.append(resolutions_item)

        element_class = d.pop("elementClass")

        dataset_info_response_200_data_source_data_layers_item = cls(
            name=name,
            category=category,
            bounding_box=bounding_box,
            resolutions=resolutions,
            element_class=element_class,
        )

        dataset_info_response_200_data_source_data_layers_item.additional_properties = d
        return dataset_info_response_200_data_source_data_layers_item

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

from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetInfoResponse200DataSourceDataLayersItemBoundingBox")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSourceDataLayersItemBoundingBox:
    """ """

    top_left: Union[Unset, List[int]] = UNSET
    width: Union[Unset, int] = UNSET
    height: Union[Unset, int] = UNSET
    depth: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        top_left: Union[Unset, List[int]] = UNSET
        if not isinstance(self.top_left, Unset):
            top_left = self.top_left

        width = self.width
        height = self.height
        depth = self.depth

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if top_left is not UNSET:
            field_dict["topLeft"] = top_left
        if width is not UNSET:
            field_dict["width"] = width
        if height is not UNSET:
            field_dict["height"] = height
        if depth is not UNSET:
            field_dict["depth"] = depth

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        top_left = cast(List[int], d.pop("topLeft", UNSET))

        width = d.pop("width", UNSET)

        height = d.pop("height", UNSET)

        depth = d.pop("depth", UNSET)

        dataset_info_response_200_data_source_data_layers_item_bounding_box = cls(
            top_left=top_left,
            width=width,
            height=height,
            depth=depth,
        )

        dataset_info_response_200_data_source_data_layers_item_bounding_box.additional_properties = (
            d
        )
        return dataset_info_response_200_data_source_data_layers_item_bounding_box

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

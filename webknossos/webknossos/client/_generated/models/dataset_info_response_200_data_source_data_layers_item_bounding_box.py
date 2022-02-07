from typing import Any, Dict, List, Type, TypeVar, cast

import attr

T = TypeVar("T", bound="DatasetInfoResponse200DataSourceDataLayersItemBoundingBox")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSourceDataLayersItemBoundingBox:
    """ """

    top_left: List[int]
    width: int
    height: int
    depth: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        top_left = self.top_left

        width = self.width
        height = self.height
        depth = self.depth

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "topLeft": top_left,
                "width": width,
                "height": height,
                "depth": depth,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        top_left = cast(List[int], d.pop("topLeft"))

        width = d.pop("width")

        height = d.pop("height")

        depth = d.pop("depth")

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

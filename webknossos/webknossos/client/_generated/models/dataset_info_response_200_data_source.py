from typing import Any, Dict, List, Type, TypeVar, cast

import attr

from ..models.dataset_info_response_200_data_source_data_layers_item import (
    DatasetInfoResponse200DataSourceDataLayersItem,
)
from ..models.dataset_info_response_200_data_source_id import (
    DatasetInfoResponse200DataSourceId,
)

T = TypeVar("T", bound="DatasetInfoResponse200DataSource")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSource:
    """ """

    id: DatasetInfoResponse200DataSourceId
    data_layers: List[DatasetInfoResponse200DataSourceDataLayersItem]
    scale: List[int]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id.to_dict()

        data_layers = []
        for data_layers_item_data in self.data_layers:
            data_layers_item = data_layers_item_data.to_dict()

            data_layers.append(data_layers_item)

        scale = self.scale

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "dataLayers": data_layers,
                "scale": scale,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = DatasetInfoResponse200DataSourceId.from_dict(d.pop("id"))

        data_layers = []
        _data_layers = d.pop("dataLayers")
        for data_layers_item_data in _data_layers:
            data_layers_item = DatasetInfoResponse200DataSourceDataLayersItem.from_dict(
                data_layers_item_data
            )

            data_layers.append(data_layers_item)

        scale = cast(List[int], d.pop("scale"))

        dataset_info_response_200_data_source = cls(
            id=id,
            data_layers=data_layers,
            scale=scale,
        )

        dataset_info_response_200_data_source.additional_properties = d
        return dataset_info_response_200_data_source

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

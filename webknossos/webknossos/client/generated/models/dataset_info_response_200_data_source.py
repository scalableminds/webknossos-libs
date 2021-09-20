from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..models.dataset_info_response_200_data_source_data_layers_item import (
    DatasetInfoResponse200DataSourceDataLayersItem,
)
from ..models.dataset_info_response_200_data_source_id import (
    DatasetInfoResponse200DataSourceId,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="DatasetInfoResponse200DataSource")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSource:
    """ """

    id: Union[Unset, DatasetInfoResponse200DataSourceId] = UNSET
    data_layers: Union[
        Unset, List[DatasetInfoResponse200DataSourceDataLayersItem]
    ] = UNSET
    scale: Union[Unset, List[float]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.id, Unset):
            id = self.id.to_dict()

        data_layers: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.data_layers, Unset):
            data_layers = []
            for data_layers_item_data in self.data_layers:
                data_layers_item = data_layers_item_data.to_dict()

                data_layers.append(data_layers_item)

        scale: Union[Unset, List[float]] = UNSET
        if not isinstance(self.scale, Unset):
            scale = self.scale

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if data_layers is not UNSET:
            field_dict["dataLayers"] = data_layers
        if scale is not UNSET:
            field_dict["scale"] = scale

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        _id = d.pop("id", UNSET)
        id: Union[Unset, DatasetInfoResponse200DataSourceId]
        if isinstance(_id, Unset):
            id = UNSET
        else:
            id = DatasetInfoResponse200DataSourceId.from_dict(_id)

        data_layers = []
        _data_layers = d.pop("dataLayers", UNSET)
        for data_layers_item_data in _data_layers or []:
            data_layers_item = DatasetInfoResponse200DataSourceDataLayersItem.from_dict(
                data_layers_item_data
            )

            data_layers.append(data_layers_item)

        scale = cast(List[float], d.pop("scale", UNSET))

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

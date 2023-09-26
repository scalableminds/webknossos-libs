from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="LinkedLayerIdentifier")


@attr.s(auto_attribs=True)
class LinkedLayerIdentifier:
    """
    Attributes:
        organization_name (str):
        data_set_name (str):
        layer_name (str):
        new_layer_name (Union[Unset, str]):
    """

    organization_name: str
    data_set_name: str
    layer_name: str
    new_layer_name: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        organization_name = self.organization_name
        data_set_name = self.data_set_name
        layer_name = self.layer_name
        new_layer_name = self.new_layer_name

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "organizationName": organization_name,
                "dataSetName": data_set_name,
                "layerName": layer_name,
            }
        )
        if new_layer_name is not UNSET:
            field_dict["newLayerName"] = new_layer_name

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        organization_name = d.pop("organizationName")

        data_set_name = d.pop("dataSetName")

        layer_name = d.pop("layerName")

        new_layer_name = d.pop("newLayerName", UNSET)

        linked_layer_identifier = cls(
            organization_name=organization_name,
            data_set_name=data_set_name,
            layer_name=layer_name,
            new_layer_name=new_layer_name,
        )

        linked_layer_identifier.additional_properties = d
        return linked_layer_identifier

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

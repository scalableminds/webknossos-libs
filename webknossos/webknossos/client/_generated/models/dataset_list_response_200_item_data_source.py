from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dataset_list_response_200_item_data_source_id import (
        DatasetListResponse200ItemDataSourceId,
    )


T = TypeVar("T", bound="DatasetListResponse200ItemDataSource")


@attr.s(auto_attribs=True)
class DatasetListResponse200ItemDataSource:
    """
    Attributes:
        id (DatasetListResponse200ItemDataSourceId):
        status (Union[Unset, str]):
    """

    id: "DatasetListResponse200ItemDataSourceId"
    status: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id.to_dict()

        status = self.status

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
            }
        )
        if status is not UNSET:
            field_dict["status"] = status

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.dataset_list_response_200_item_data_source_id import (
            DatasetListResponse200ItemDataSourceId,
        )

        d = src_dict.copy()
        id = DatasetListResponse200ItemDataSourceId.from_dict(d.pop("id"))

        status = d.pop("status", UNSET)

        dataset_list_response_200_item_data_source = cls(
            id=id,
            status=status,
        )

        dataset_list_response_200_item_data_source.additional_properties = d
        return dataset_list_response_200_item_data_source

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

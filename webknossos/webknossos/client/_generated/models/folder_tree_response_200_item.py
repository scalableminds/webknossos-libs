from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="FolderTreeResponse200Item")


@attr.s(auto_attribs=True)
class FolderTreeResponse200Item:
    """
    Attributes:
        id (str):
        name (str):
        parent (Union[Unset, str]):
        is_editable (Union[Unset, int]):
    """

    id: str
    name: str
    parent: Union[Unset, str] = UNSET
    is_editable: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        name = self.name
        parent = self.parent
        is_editable = self.is_editable

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
            }
        )
        if parent is not UNSET:
            field_dict["parent"] = parent
        if is_editable is not UNSET:
            field_dict["isEditable"] = is_editable

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        name = d.pop("name")

        parent = d.pop("parent", UNSET)

        is_editable = d.pop("isEditable", UNSET)

        folder_tree_response_200_item = cls(
            id=id,
            name=name,
            parent=parent,
            is_editable=is_editable,
        )

        folder_tree_response_200_item.additional_properties = d
        return folder_tree_response_200_item

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

from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.linked_layer_identifier import LinkedLayerIdentifier


T = TypeVar("T", bound="ReserveUploadInformation")


@attr.s(auto_attribs=True)
class ReserveUploadInformation:
    """
    Attributes:
        upload_id (str):
        name (str):
        organization (str):
        total_file_count (int):
        initial_teams (List[str]):
        layers_to_link (Union[Unset, List['LinkedLayerIdentifier']]):
        folder_id (Union[Unset, str]):
    """

    upload_id: str
    name: str
    organization: str
    total_file_count: int
    initial_teams: List[str]
    layers_to_link: Union[Unset, List["LinkedLayerIdentifier"]] = UNSET
    folder_id: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        upload_id = self.upload_id
        name = self.name
        organization = self.organization
        total_file_count = self.total_file_count
        initial_teams = self.initial_teams

        layers_to_link: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.layers_to_link, Unset):
            layers_to_link = []
            for layers_to_link_item_data in self.layers_to_link:
                layers_to_link_item = layers_to_link_item_data.to_dict()

                layers_to_link.append(layers_to_link_item)

        folder_id = self.folder_id

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "uploadId": upload_id,
                "name": name,
                "organization": organization,
                "totalFileCount": total_file_count,
                "initialTeams": initial_teams,
            }
        )
        if layers_to_link is not UNSET:
            field_dict["layersToLink"] = layers_to_link
        if folder_id is not UNSET:
            field_dict["folderId"] = folder_id

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.linked_layer_identifier import LinkedLayerIdentifier

        d = src_dict.copy()
        upload_id = d.pop("uploadId")

        name = d.pop("name")

        organization = d.pop("organization")

        total_file_count = d.pop("totalFileCount")

        initial_teams = cast(List[str], d.pop("initialTeams"))

        layers_to_link = []
        _layers_to_link = d.pop("layersToLink", UNSET)
        for layers_to_link_item_data in _layers_to_link or []:
            layers_to_link_item = LinkedLayerIdentifier.from_dict(
                layers_to_link_item_data
            )

            layers_to_link.append(layers_to_link_item)

        folder_id = d.pop("folderId", UNSET)

        reserve_upload_information = cls(
            upload_id=upload_id,
            name=name,
            organization=organization,
            total_file_count=total_file_count,
            initial_teams=initial_teams,
            layers_to_link=layers_to_link,
            folder_id=folder_id,
        )

        reserve_upload_information.additional_properties = d
        return reserve_upload_information

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

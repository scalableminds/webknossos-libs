from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="DatasetInfoResponse200DataSourceId")


@attr.s(auto_attribs=True)
class DatasetInfoResponse200DataSourceId:
    """ """

    name: str
    team: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        team = self.team

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "team": team,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        team = d.pop("team")

        dataset_info_response_200_data_source_id = cls(
            name=name,
            team=team,
        )

        dataset_info_response_200_data_source_id.additional_properties = d
        return dataset_info_response_200_data_source_id

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

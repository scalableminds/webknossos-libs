from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="BuildInfoResponse200WebknossosWrap")


@attr.s(auto_attribs=True)
class BuildInfoResponse200WebknossosWrap:
    """ """

    built_at_millis: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    commit_hash: Union[Unset, str] = UNSET
    scala_version: Union[Unset, str] = UNSET
    version: Union[Unset, str] = UNSET
    sbt_version: Union[Unset, str] = UNSET
    built_at_string: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        built_at_millis = self.built_at_millis
        name = self.name
        commit_hash = self.commit_hash
        scala_version = self.scala_version
        version = self.version
        sbt_version = self.sbt_version
        built_at_string = self.built_at_string

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if built_at_millis is not UNSET:
            field_dict["builtAtMillis"] = built_at_millis
        if name is not UNSET:
            field_dict["name"] = name
        if commit_hash is not UNSET:
            field_dict["commitHash"] = commit_hash
        if scala_version is not UNSET:
            field_dict["scalaVersion"] = scala_version
        if version is not UNSET:
            field_dict["version"] = version
        if sbt_version is not UNSET:
            field_dict["sbtVersion"] = sbt_version
        if built_at_string is not UNSET:
            field_dict["builtAtString"] = built_at_string

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        built_at_millis = d.pop("builtAtMillis", UNSET)

        name = d.pop("name", UNSET)

        commit_hash = d.pop("commitHash", UNSET)

        scala_version = d.pop("scalaVersion", UNSET)

        version = d.pop("version", UNSET)

        sbt_version = d.pop("sbtVersion", UNSET)

        built_at_string = d.pop("builtAtString", UNSET)

        build_info_response_200_webknossos_wrap = cls(
            built_at_millis=built_at_millis,
            name=name,
            commit_hash=commit_hash,
            scala_version=scala_version,
            version=version,
            sbt_version=sbt_version,
            built_at_string=built_at_string,
        )

        build_info_response_200_webknossos_wrap.additional_properties = d
        return build_info_response_200_webknossos_wrap

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

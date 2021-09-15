from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="BuildInfoResponse200Webknossos")


@attr.s(auto_attribs=True)
class BuildInfoResponse200Webknossos:
    """ """

    name: Union[Unset, str] = UNSET
    ci_tag: Union[Unset, str] = UNSET
    commit_hash: Union[Unset, str] = UNSET
    ci_build: Union[Unset, str] = UNSET
    scala_version: Union[Unset, str] = UNSET
    version: Union[Unset, str] = UNSET
    sbt_version: Union[Unset, str] = UNSET
    datastore_api_version: Union[Unset, str] = UNSET
    commit_date: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        ci_tag = self.ci_tag
        commit_hash = self.commit_hash
        ci_build = self.ci_build
        scala_version = self.scala_version
        version = self.version
        sbt_version = self.sbt_version
        datastore_api_version = self.datastore_api_version
        commit_date = self.commit_date

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if ci_tag is not UNSET:
            field_dict["ciTag"] = ci_tag
        if commit_hash is not UNSET:
            field_dict["commitHash"] = commit_hash
        if ci_build is not UNSET:
            field_dict["ciBuild"] = ci_build
        if scala_version is not UNSET:
            field_dict["scalaVersion"] = scala_version
        if version is not UNSET:
            field_dict["version"] = version
        if sbt_version is not UNSET:
            field_dict["sbtVersion"] = sbt_version
        if datastore_api_version is not UNSET:
            field_dict["datastoreApiVersion"] = datastore_api_version
        if commit_date is not UNSET:
            field_dict["commitDate"] = commit_date

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        ci_tag = d.pop("ciTag", UNSET)

        commit_hash = d.pop("commitHash", UNSET)

        ci_build = d.pop("ciBuild", UNSET)

        scala_version = d.pop("scalaVersion", UNSET)

        version = d.pop("version", UNSET)

        sbt_version = d.pop("sbtVersion", UNSET)

        datastore_api_version = d.pop("datastoreApiVersion", UNSET)

        commit_date = d.pop("commitDate", UNSET)

        build_info_response_200_webknossos = cls(
            name=name,
            ci_tag=ci_tag,
            commit_hash=commit_hash,
            ci_build=ci_build,
            scala_version=scala_version,
            version=version,
            sbt_version=sbt_version,
            datastore_api_version=datastore_api_version,
            commit_date=commit_date,
        )

        build_info_response_200_webknossos.additional_properties = d
        return build_info_response_200_webknossos

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

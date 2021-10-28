from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="BuildInfoResponse200Webknossos")


@attr.s(auto_attribs=True)
class BuildInfoResponse200Webknossos:
    """ """

    name: str
    ci_tag: str
    commit_hash: str
    ci_build: str
    scala_version: str
    version: str
    sbt_version: str
    datastore_api_version: str
    commit_date: str
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
        field_dict.update(
            {
                "name": name,
                "ciTag": ci_tag,
                "commitHash": commit_hash,
                "ciBuild": ci_build,
                "scalaVersion": scala_version,
                "version": version,
                "sbtVersion": sbt_version,
                "datastoreApiVersion": datastore_api_version,
                "commitDate": commit_date,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        ci_tag = d.pop("ciTag")

        commit_hash = d.pop("commitHash")

        ci_build = d.pop("ciBuild")

        scala_version = d.pop("scalaVersion")

        version = d.pop("version")

        sbt_version = d.pop("sbtVersion")

        datastore_api_version = d.pop("datastoreApiVersion")

        commit_date = d.pop("commitDate")

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

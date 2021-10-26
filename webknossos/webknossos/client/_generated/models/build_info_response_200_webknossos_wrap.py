from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="BuildInfoResponse200WebknossosWrap")


@attr.s(auto_attribs=True)
class BuildInfoResponse200WebknossosWrap:
    """ """

    built_at_millis: str
    name: str
    commit_hash: str
    scala_version: str
    version: str
    sbt_version: str
    built_at_string: str
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
        field_dict.update(
            {
                "builtAtMillis": built_at_millis,
                "name": name,
                "commitHash": commit_hash,
                "scalaVersion": scala_version,
                "version": version,
                "sbtVersion": sbt_version,
                "builtAtString": built_at_string,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        built_at_millis = d.pop("builtAtMillis")

        name = d.pop("name")

        commit_hash = d.pop("commitHash")

        scala_version = d.pop("scalaVersion")

        version = d.pop("version")

        sbt_version = d.pop("sbtVersion")

        built_at_string = d.pop("builtAtString")

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

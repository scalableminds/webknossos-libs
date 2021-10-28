from typing import Any, Dict, List, Type, TypeVar

import attr

from ..models.build_info_response_200_webknossos import BuildInfoResponse200Webknossos
from ..models.build_info_response_200_webknossos_wrap import (
    BuildInfoResponse200WebknossosWrap,
)

T = TypeVar("T", bound="BuildInfoResponse200")


@attr.s(auto_attribs=True)
class BuildInfoResponse200:
    """ """

    webknossos: BuildInfoResponse200Webknossos
    webknossos_wrap: BuildInfoResponse200WebknossosWrap
    schema_version: int
    token: str
    local_data_store_enabled: int
    local_tracing_store_enabled: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        webknossos = self.webknossos.to_dict()

        webknossos_wrap = self.webknossos_wrap.to_dict()

        schema_version = self.schema_version
        token = self.token
        local_data_store_enabled = self.local_data_store_enabled
        local_tracing_store_enabled = self.local_tracing_store_enabled

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "webknossos": webknossos,
                "webknossos-wrap": webknossos_wrap,
                "schemaVersion": schema_version,
                "token": token,
                "localDataStoreEnabled": local_data_store_enabled,
                "localTracingStoreEnabled": local_tracing_store_enabled,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        webknossos = BuildInfoResponse200Webknossos.from_dict(d.pop("webknossos"))

        webknossos_wrap = BuildInfoResponse200WebknossosWrap.from_dict(
            d.pop("webknossos-wrap")
        )

        schema_version = d.pop("schemaVersion")

        token = d.pop("token")

        local_data_store_enabled = d.pop("localDataStoreEnabled")

        local_tracing_store_enabled = d.pop("localTracingStoreEnabled")

        build_info_response_200 = cls(
            webknossos=webknossos,
            webknossos_wrap=webknossos_wrap,
            schema_version=schema_version,
            token=token,
            local_data_store_enabled=local_data_store_enabled,
            local_tracing_store_enabled=local_tracing_store_enabled,
        )

        build_info_response_200.additional_properties = d
        return build_info_response_200

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

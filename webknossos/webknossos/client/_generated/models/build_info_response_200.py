from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.build_info_response_200_webknossos import (
        BuildInfoResponse200Webknossos,
    )
    from ..models.build_info_response_200_webknossos_wrap import (
        BuildInfoResponse200WebknossosWrap,
    )


T = TypeVar("T", bound="BuildInfoResponse200")


@attr.s(auto_attribs=True)
class BuildInfoResponse200:
    """
    Attributes:
        webknossos (Union[Unset, BuildInfoResponse200Webknossos]):
        webknossos_wrap (Union[Unset, BuildInfoResponse200WebknossosWrap]):
        schema_version (Union[Unset, int]):
        local_data_store_enabled (Union[Unset, int]):
        local_tracing_store_enabled (Union[Unset, int]):
    """

    webknossos: Union[Unset, "BuildInfoResponse200Webknossos"] = UNSET
    webknossos_wrap: Union[Unset, "BuildInfoResponse200WebknossosWrap"] = UNSET
    schema_version: Union[Unset, int] = UNSET
    local_data_store_enabled: Union[Unset, int] = UNSET
    local_tracing_store_enabled: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        webknossos: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.webknossos, Unset):
            webknossos = self.webknossos.to_dict()

        webknossos_wrap: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.webknossos_wrap, Unset):
            webknossos_wrap = self.webknossos_wrap.to_dict()

        schema_version = self.schema_version
        local_data_store_enabled = self.local_data_store_enabled
        local_tracing_store_enabled = self.local_tracing_store_enabled

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if webknossos is not UNSET:
            field_dict["webknossos"] = webknossos
        if webknossos_wrap is not UNSET:
            field_dict["webknossos-wrap"] = webknossos_wrap
        if schema_version is not UNSET:
            field_dict["schemaVersion"] = schema_version
        if local_data_store_enabled is not UNSET:
            field_dict["localDataStoreEnabled"] = local_data_store_enabled
        if local_tracing_store_enabled is not UNSET:
            field_dict["localTracingStoreEnabled"] = local_tracing_store_enabled

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.build_info_response_200_webknossos import (
            BuildInfoResponse200Webknossos,
        )
        from ..models.build_info_response_200_webknossos_wrap import (
            BuildInfoResponse200WebknossosWrap,
        )

        d = src_dict.copy()
        _webknossos = d.pop("webknossos", UNSET)
        webknossos: Union[Unset, BuildInfoResponse200Webknossos]
        if isinstance(_webknossos, Unset):
            webknossos = UNSET
        else:
            webknossos = BuildInfoResponse200Webknossos.from_dict(_webknossos)

        _webknossos_wrap = d.pop("webknossos-wrap", UNSET)
        webknossos_wrap: Union[Unset, BuildInfoResponse200WebknossosWrap]
        if isinstance(_webknossos_wrap, Unset):
            webknossos_wrap = UNSET
        else:
            webknossos_wrap = BuildInfoResponse200WebknossosWrap.from_dict(
                _webknossos_wrap
            )

        schema_version = d.pop("schemaVersion", UNSET)

        local_data_store_enabled = d.pop("localDataStoreEnabled", UNSET)

        local_tracing_store_enabled = d.pop("localTracingStoreEnabled", UNSET)

        build_info_response_200 = cls(
            webknossos=webknossos,
            webknossos_wrap=webknossos_wrap,
            schema_version=schema_version,
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

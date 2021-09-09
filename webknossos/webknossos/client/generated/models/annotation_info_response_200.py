from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..models.annotation_info_response_200_data_store import (
    AnnotationInfoResponse200DataStore,
)
from ..models.annotation_info_response_200_restrictions import (
    AnnotationInfoResponse200Restrictions,
)
from ..models.annotation_info_response_200_settings import (
    AnnotationInfoResponse200Settings,
)
from ..models.annotation_info_response_200_stats import AnnotationInfoResponse200Stats
from ..models.annotation_info_response_200_tracing import (
    AnnotationInfoResponse200Tracing,
)
from ..models.annotation_info_response_200_tracing_store import (
    AnnotationInfoResponse200TracingStore,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationInfoResponse200")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200:
    """ """

    modified: Union[Unset, int] = UNSET
    state: Union[Unset, str] = UNSET
    id: Union[Unset, str] = UNSET
    name: Union[Unset, str] = UNSET
    description: Union[Unset, str] = UNSET
    typ: Union[Unset, str] = UNSET
    task: Union[Unset, str] = UNSET
    stats: Union[Unset, AnnotationInfoResponse200Stats] = UNSET
    restrictions: Union[Unset, AnnotationInfoResponse200Restrictions] = UNSET
    formatted_hash: Union[Unset, str] = UNSET
    tracing: Union[Unset, AnnotationInfoResponse200Tracing] = UNSET
    data_set_name: Union[Unset, str] = UNSET
    organization: Union[Unset, str] = UNSET
    data_store: Union[Unset, AnnotationInfoResponse200DataStore] = UNSET
    tracing_store: Union[Unset, AnnotationInfoResponse200TracingStore] = UNSET
    visibility: Union[Unset, str] = UNSET
    settings: Union[Unset, AnnotationInfoResponse200Settings] = UNSET
    tracing_time: Union[Unset, int] = UNSET
    tags: Union[Unset, List[str]] = UNSET
    user: Union[Unset, str] = UNSET
    meshes: Union[Unset, List[Any]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        modified = self.modified
        state = self.state
        id = self.id
        name = self.name
        description = self.description
        typ = self.typ
        task = self.task
        stats: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.stats, Unset):
            stats = self.stats.to_dict()

        restrictions: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.restrictions, Unset):
            restrictions = self.restrictions.to_dict()

        formatted_hash = self.formatted_hash
        tracing: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.tracing, Unset):
            tracing = self.tracing.to_dict()

        data_set_name = self.data_set_name
        organization = self.organization
        data_store: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.data_store, Unset):
            data_store = self.data_store.to_dict()

        tracing_store: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.tracing_store, Unset):
            tracing_store = self.tracing_store.to_dict()

        visibility = self.visibility
        settings: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.settings, Unset):
            settings = self.settings.to_dict()

        tracing_time = self.tracing_time
        tags: Union[Unset, List[str]] = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        user = self.user
        meshes: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.meshes, Unset):
            meshes = []
            for meshes_item_data in self.meshes:
                meshes_item = meshes_item_data

                meshes.append(meshes_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if modified is not UNSET:
            field_dict["modified"] = modified
        if state is not UNSET:
            field_dict["state"] = state
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name
        if description is not UNSET:
            field_dict["description"] = description
        if typ is not UNSET:
            field_dict["typ"] = typ
        if task is not UNSET:
            field_dict["task"] = task
        if stats is not UNSET:
            field_dict["stats"] = stats
        if restrictions is not UNSET:
            field_dict["restrictions"] = restrictions
        if formatted_hash is not UNSET:
            field_dict["formattedHash"] = formatted_hash
        if tracing is not UNSET:
            field_dict["tracing"] = tracing
        if data_set_name is not UNSET:
            field_dict["dataSetName"] = data_set_name
        if organization is not UNSET:
            field_dict["organization"] = organization
        if data_store is not UNSET:
            field_dict["dataStore"] = data_store
        if tracing_store is not UNSET:
            field_dict["tracingStore"] = tracing_store
        if visibility is not UNSET:
            field_dict["visibility"] = visibility
        if settings is not UNSET:
            field_dict["settings"] = settings
        if tracing_time is not UNSET:
            field_dict["tracingTime"] = tracing_time
        if tags is not UNSET:
            field_dict["tags"] = tags
        if user is not UNSET:
            field_dict["user"] = user
        if meshes is not UNSET:
            field_dict["meshes"] = meshes

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        modified = d.pop("modified", UNSET)

        state = d.pop("state", UNSET)

        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        description = d.pop("description", UNSET)

        typ = d.pop("typ", UNSET)

        task = d.pop("task", UNSET)

        _stats = d.pop("stats", UNSET)
        stats: Union[Unset, AnnotationInfoResponse200Stats]
        if isinstance(_stats, Unset):
            stats = UNSET
        else:
            stats = AnnotationInfoResponse200Stats.from_dict(_stats)

        _restrictions = d.pop("restrictions", UNSET)
        restrictions: Union[Unset, AnnotationInfoResponse200Restrictions]
        if isinstance(_restrictions, Unset):
            restrictions = UNSET
        else:
            restrictions = AnnotationInfoResponse200Restrictions.from_dict(
                _restrictions
            )

        formatted_hash = d.pop("formattedHash", UNSET)

        _tracing = d.pop("tracing", UNSET)
        tracing: Union[Unset, AnnotationInfoResponse200Tracing]
        if isinstance(_tracing, Unset):
            tracing = UNSET
        else:
            tracing = AnnotationInfoResponse200Tracing.from_dict(_tracing)

        data_set_name = d.pop("dataSetName", UNSET)

        organization = d.pop("organization", UNSET)

        _data_store = d.pop("dataStore", UNSET)
        data_store: Union[Unset, AnnotationInfoResponse200DataStore]
        if isinstance(_data_store, Unset):
            data_store = UNSET
        else:
            data_store = AnnotationInfoResponse200DataStore.from_dict(_data_store)

        _tracing_store = d.pop("tracingStore", UNSET)
        tracing_store: Union[Unset, AnnotationInfoResponse200TracingStore]
        if isinstance(_tracing_store, Unset):
            tracing_store = UNSET
        else:
            tracing_store = AnnotationInfoResponse200TracingStore.from_dict(
                _tracing_store
            )

        visibility = d.pop("visibility", UNSET)

        _settings = d.pop("settings", UNSET)
        settings: Union[Unset, AnnotationInfoResponse200Settings]
        if isinstance(_settings, Unset):
            settings = UNSET
        else:
            settings = AnnotationInfoResponse200Settings.from_dict(_settings)

        tracing_time = d.pop("tracingTime", UNSET)

        tags = cast(List[str], d.pop("tags", UNSET))

        user = d.pop("user", UNSET)

        meshes = []
        _meshes = d.pop("meshes", UNSET)
        for meshes_item_data in _meshes or []:
            meshes_item = meshes_item_data

            meshes.append(meshes_item)

        annotation_info_response_200 = cls(
            modified=modified,
            state=state,
            id=id,
            name=name,
            description=description,
            typ=typ,
            task=task,
            stats=stats,
            restrictions=restrictions,
            formatted_hash=formatted_hash,
            tracing=tracing,
            data_set_name=data_set_name,
            organization=organization,
            data_store=data_store,
            tracing_store=tracing_store,
            visibility=visibility,
            settings=settings,
            tracing_time=tracing_time,
            tags=tags,
            user=user,
            meshes=meshes,
        )

        annotation_info_response_200.additional_properties = d
        return annotation_info_response_200

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

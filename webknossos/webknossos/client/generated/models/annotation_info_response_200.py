from typing import Any, Dict, List, Type, TypeVar, cast

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
from ..models.annotation_info_response_200_user import AnnotationInfoResponse200User

T = TypeVar("T", bound="AnnotationInfoResponse200")


@attr.s(auto_attribs=True)
class AnnotationInfoResponse200:
    """ """

    modified: int
    state: str
    id: str
    name: str
    description: str
    typ: str
    task: str
    stats: AnnotationInfoResponse200Stats
    restrictions: AnnotationInfoResponse200Restrictions
    formatted_hash: str
    tracing: AnnotationInfoResponse200Tracing
    data_set_name: str
    organization: str
    data_store: AnnotationInfoResponse200DataStore
    tracing_store: AnnotationInfoResponse200TracingStore
    visibility: str
    settings: AnnotationInfoResponse200Settings
    tracing_time: int
    tags: List[str]
    user: AnnotationInfoResponse200User
    meshes: List[Any]
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        modified = self.modified
        state = self.state
        id = self.id
        name = self.name
        description = self.description
        typ = self.typ
        task = self.task
        stats = self.stats.to_dict()

        restrictions = self.restrictions.to_dict()

        formatted_hash = self.formatted_hash
        tracing = self.tracing.to_dict()

        data_set_name = self.data_set_name
        organization = self.organization
        data_store = self.data_store.to_dict()

        tracing_store = self.tracing_store.to_dict()

        visibility = self.visibility
        settings = self.settings.to_dict()

        tracing_time = self.tracing_time
        tags = self.tags

        user = self.user.to_dict()

        meshes = []
        for meshes_item_data in self.meshes:
            meshes_item = meshes_item_data

            meshes.append(meshes_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "modified": modified,
                "state": state,
                "id": id,
                "name": name,
                "description": description,
                "typ": typ,
                "task": task,
                "stats": stats,
                "restrictions": restrictions,
                "formattedHash": formatted_hash,
                "tracing": tracing,
                "dataSetName": data_set_name,
                "organization": organization,
                "dataStore": data_store,
                "tracingStore": tracing_store,
                "visibility": visibility,
                "settings": settings,
                "tracingTime": tracing_time,
                "tags": tags,
                "user": user,
                "meshes": meshes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        modified = d.pop("modified")

        state = d.pop("state")

        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description")

        typ = d.pop("typ")

        task = d.pop("task")

        stats = AnnotationInfoResponse200Stats.from_dict(d.pop("stats"))

        restrictions = AnnotationInfoResponse200Restrictions.from_dict(
            d.pop("restrictions")
        )

        formatted_hash = d.pop("formattedHash")

        tracing = AnnotationInfoResponse200Tracing.from_dict(d.pop("tracing"))

        data_set_name = d.pop("dataSetName")

        organization = d.pop("organization")

        data_store = AnnotationInfoResponse200DataStore.from_dict(d.pop("dataStore"))

        tracing_store = AnnotationInfoResponse200TracingStore.from_dict(
            d.pop("tracingStore")
        )

        visibility = d.pop("visibility")

        settings = AnnotationInfoResponse200Settings.from_dict(d.pop("settings"))

        tracing_time = d.pop("tracingTime")

        tags = cast(List[str], d.pop("tags"))

        user = AnnotationInfoResponse200User.from_dict(d.pop("user"))

        meshes = []
        _meshes = d.pop("meshes")
        for meshes_item_data in _meshes:
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
